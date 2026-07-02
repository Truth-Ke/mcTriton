#include "Gluon/GluonC500AsyncCopyLayout.h"
#include "Gluon/GluonC500LayoutHelpers.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon::c500 {

static Attribute unwrapNoVerifyEncoding(Attribute attr) {
  if (auto noVerify =
          dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(attr))
    return noVerify.getLayout();
  return attr;
}

static Type cloneTypeWithEncoding(Type type, Attribute encoding) {
  encoding = unwrapNoVerifyEncoding(encoding);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorTy.getShape(),
                                 tensorTy.getElementType(), encoding);
  if (auto memdescTy = dyn_cast<ttg::MemDescType>(type))
    return ttg::MemDescType::get(
        memdescTy.getShape(), memdescTy.getElementType(), encoding,
        memdescTy.getMemorySpace(), memdescTy.getMutableMemory(),
        memdescTy.getAllocShape());
  return type;
}

static bool hasAutoEncoding(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  return tensorTy &&
         isa<gluon_dialect::AutoEncodingAttr>(tensorTy.getEncoding());
}

bool isFedByC500AsyncCopyProducer(Value memdesc) {
  return isFedByAnyMemDescUser<ttg::AsyncCopyGlobalToLocalOp>(memdesc);
}

static Attribute computeSharedEncodingFromDotConsumer(
    const DotConsumerInfo &info, ArrayRef<unsigned> order, MLIRContext *ctx) {
  auto ctaLayout = ttg::getCTALayout(info.dotEnc);
  return ttg::SwizzledSharedEncodingAttr::get(
      ctx, info.dotEnc, info.tensorShape, order, ctaLayout, info.elementType,
      /*needTrans=*/false);
}

static Attribute getDefaultAsyncCopySharedEncoding(
    ttg::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> order) {
  auto dstTy = cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  auto ctaLayout =
      ttg::CTAEncodingAttr::getDefault(copyOp.getContext(), dstTy.getRank());
  return ttg::SwizzledSharedEncodingAttr::get(copyOp.getContext(), 1, 1, 1,
                                              order, ctaLayout);
}

FailureOr<Attribute> chooseAsyncCopySharedEncoding(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Value dstMemDesc,
    std::optional<DotConsumerInfo> dotConsumer) {
  SmallVector<unsigned, 2> srcOrder =
      chooseRank2ContiguousOrder(copyOp.getSrc());
  SmallVector<std::pair<Attribute, StringRef>> candidates;

  if (dotConsumer) {
    candidates.push_back({computeSharedEncodingFromDotConsumer(
                              *dotConsumer, srcOrder, copyOp.getContext()),
                          "dot-derived"});
    if (dotConsumer->userSharedOrder &&
        ArrayRef<unsigned>(*dotConsumer->userSharedOrder) !=
            ArrayRef<unsigned>(srcOrder)) {
      candidates.push_back({computeSharedEncodingFromDotConsumer(
                                *dotConsumer, *dotConsumer->userSharedOrder,
                                copyOp.getContext()),
                            "user-order-preserving"});
    }
  } else {
    candidates.push_back({getDefaultAsyncCopySharedEncoding(copyOp, srcOrder),
                          "c500-default"});
  }

  for (auto [candidate, provenance] : candidates) {
    (void)provenance;
    if (isC500AsyncCopyContiguousSharedWrite(copyOp, candidate))
      return candidate;
  }

  InFlightDiagnostic diag = copyOp.emitError()
                            << "failed to choose a C500 async_copy shared "
                               "encoding compatible with both downstream dot "
                               "layout and contiguous shared writes";
  if (dotConsumer)
    diag << "; downstream dot operand encoding is " << dotConsumer->dotEnc;
  diag << "; destination memdesc is " << dstMemDesc;
  return failure();
}

LogicalResult materializeAsyncCopyFollowerLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
    auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
    if (!srcTy || !srcTy.getEncoding() || hasAutoEncoding(copyOp.getSrc()))
      return WalkResult::advance();
    Attribute srcEncoding = unwrapNoVerifyEncoding(srcTy.getEncoding());
    OpBuilder builder(copyOp);

    auto convertOperand = [&](Value operand,
                              MutableOperandRange mutableOperand) {
      auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTy || unwrapNoVerifyEncoding(operandTy.getEncoding()) ==
                            srcEncoding)
        return;
      auto targetTy =
          cast<RankedTensorType>(cloneTypeWithEncoding(operandTy, srcEncoding));
      auto converted = builder.create<ttg::ConvertLayoutOp>(
          copyOp.getLoc(), targetTy, operand);
      mutableOperand.assign(converted.getResult());
    };

    if (copyOp.getMask())
      convertOperand(copyOp.getMask(), copyOp.getMaskMutable());
    if (copyOp.getOther())
      convertOperand(copyOp.getOther(), copyOp.getOtherMutable());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult verifyAsyncCopyLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
    auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
    if (!srcTy || !srcTy.getEncoding()) {
      copyOp.emitError() << "async_copy source has no concrete layout";
      return WalkResult::interrupt();
    }
    Attribute srcEncoding = unwrapNoVerifyEncoding(srcTy.getEncoding());

    auto verifyFollower = [&](Value value, StringRef name) -> LogicalResult {
      if (!value)
        return success();
      auto valueTy = dyn_cast<RankedTensorType>(value.getType());
      if (!valueTy)
        return success();
      if (unwrapNoVerifyEncoding(valueTy.getEncoding()) != srcEncoding)
        return copyOp.emitError()
               << "async_copy " << name
               << " layout does not match source layout";
      return success();
    };

    if (failed(verifyFollower(copyOp.getMask(), "mask")) ||
        failed(verifyFollower(copyOp.getOther(), "other")))
      return WalkResult::interrupt();

    if (failed(verifyC500AsyncCopyContiguousSharedWrite(copyOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace mlir::triton::gpu::metax::gluon::c500
