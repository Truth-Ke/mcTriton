#include "Gluon/GluonC500LayoutHelpers.h"

#include "mlir/IR/Builders.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon::c500 {

Value getMemDescViewSource(Operation *op) {
  if (!op)
    return {};
  if (auto index = dyn_cast<ttg::MemDescIndexOp>(op))
    return index.getSrc();
  if (auto slice = dyn_cast<ttg::MemDescSubsliceOp>(op))
    return slice.getSrc();
  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op))
    return trans.getSrc();
  if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op))
    return reshape.getSrc();
  if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(op))
    return reinterpret.getSrc();
  if (auto require = dyn_cast<gluon_dialect::RequireLayoutOp>(op))
    if (isa<ttg::MemDescType>(require.getSrc().getType()))
      return require.getSrc();
  return {};
}

Value getMemDescRoot(Value value) {
  while (value) {
    Value source = getMemDescViewSource(value.getDefiningOp());
    if (!source)
      return value;
    value = source;
  }
  return {};
}

void collectMemDescAliasValues(Value root, llvm::SetVector<Value> &aliases) {
  if (!root || aliases.contains(root))
    return;
  aliases.insert(root);
  for (Operation *user : root.getUsers()) {
    if (getMemDescViewSource(user) == root && user->getNumResults() == 1)
      collectMemDescAliasValues(user->getResult(0), aliases);
  }
}

unsigned getTensorElementOrPointeeBitWidth(RankedTensorType tensorTy) {
  Type elemTy = tensorTy.getElementType();
  if (isa<tt::PointerType>(elemTy))
    return tt::getPointeeBitWidth(tensorTy);
  if (elemTy.isIntOrFloat())
    return elemTy.getIntOrFloatBitWidth();
  return 0;
}

SmallVector<unsigned, 2>
chooseRank2ContiguousOrder(Value value,
                           tt::ModuleAxisInfoAnalysis *axisInfoAnalysis) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getRank() != 2)
    return {1, 0};

  Attribute encoding = tensorTy.getEncoding();
  if (encoding && !isa<gluon_dialect::AutoEncodingAttr>(encoding)) {
    SmallVector<unsigned> order = ttg::getOrder(tensorTy);
    if (order.size() == 2)
      return {order[0], order[1]};
  }

  if (axisInfoAnalysis) {
    tt::AxisInfo *axisInfo = axisInfoAnalysis->getAxisInfo(value);
    if (axisInfo && axisInfo->getRank() == tensorTy.getRank()) {
      ArrayRef<int64_t> contiguity = axisInfo->getContiguity();
      if (contiguity.size() == 2 && contiguity[0] != contiguity[1])
        return contiguity[0] > contiguity[1] ? SmallVector<unsigned, 2>{0, 1}
                                             : SmallVector<unsigned, 2>{1, 0};
    }
  }

  ArrayRef<int64_t> shape = tensorTy.getShape();
  if (!ShapedType::isDynamic(shape[0]) && !ShapedType::isDynamic(shape[1]) &&
      shape[0] > shape[1])
    return {0, 1};
  return {1, 0};
}

std::optional<ttg::BlockedEncodingAttr>
getC500GlobalBlockedEncoding(Value value, Operation *anchor,
                             ArrayRef<unsigned> order) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return std::nullopt;

  OpBuilder builder(anchor);
  int numWarps = ttg::lookupNumWarps(anchor);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(anchor);

  if (tensorTy.getRank() == 2 && order.size() == 2) {
    unsigned elemBits = getTensorElementOrPointeeBitWidth(tensorTy);
    unsigned vecElems =
        elemBits == 0 ? 8 : std::max<unsigned>(1, 128 / elemBits);
    SmallVector<unsigned> sizePerThread(tensorTy.getRank(), 1);
    sizePerThread[order[0]] = vecElems;
    return ttg::BlockedEncodingAttr::get(
        value.getContext(), tensorTy.getShape(), sizePerThread, order,
        numWarps, threadsPerWarp, numCTAs);
  }

  return ttg::getDefaultBlockedEncoding(value.getContext(), tensorTy.getShape(),
                                        numWarps, threadsPerWarp, numCTAs);
}

std::optional<ttg::BlockedEncodingAttr>
getC500GlobalBlockedEncodingLike(Value value, Operation *anchor,
                                 ttg::BlockedEncodingAttr model) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || static_cast<int64_t>(model.getOrder().size()) !=
                       tensorTy.getRank())
    return std::nullopt;

  OpBuilder builder(anchor);
  int numWarps = ttg::lookupNumWarps(anchor);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(anchor);
  return ttg::BlockedEncodingAttr::get(
      value.getContext(), tensorTy.getShape(), model.getSizePerThread(),
      model.getOrder(), numWarps, threadsPerWarp, numCTAs);
}

static std::optional<unsigned> getDenseContiguityForDim(Attribute attr,
                                                        unsigned dim) {
  if (!attr)
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (dense.getNumElements() <= dim)
      return std::nullopt;
    auto values = dense.getValues<APInt>();
    auto it = values.begin();
    std::advance(it, dim);
    return (*it).getZExtValue();
  }
  return std::nullopt;
}

std::optional<unsigned> getContiguityHint(Value value, unsigned dim) {
  Value current = value;
  if (auto require = current.getDefiningOp<gluon_dialect::RequireLayoutOp>())
    current = require.getSrc();

  if (auto blockArg = dyn_cast<BlockArgument>(current)) {
    if (auto func =
            dyn_cast_or_null<tt::FuncOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getArgNumber() < func.getNumArguments()) {
        if (Attribute attr =
                func.getArgAttr(blockArg.getArgNumber(), "tt.contiguity"))
          return getDenseContiguityForDim(attr, dim);
      }
    }
  }
  if (Operation *def = current.getDefiningOp())
    return getDenseContiguityForDim(def->getAttr("tt.contiguity"), dim);
  return std::nullopt;
}

static bool isConsecutive(ArrayRef<int64_t> values) {
  if (values.empty())
    return false;
  for (auto [index, value] : llvm::enumerate(values))
    if (value != values.front() + static_cast<int64_t>(index))
      return false;
  return true;
}

static bool hasWholeTileGluonSlice(gluon_dialect::ExtractSliceOp extract) {
  auto sourceTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!sourceTy || !resultTy || sourceTy.getRank() != resultTy.getRank() ||
      extract.getOffsets().size() != static_cast<size_t>(sourceTy.getRank()))
    return false;

  for (auto [dim, offset] : llvm::enumerate(extract.getOffsets())) {
    int64_t sourceDim = sourceTy.getShape()[dim];
    int64_t resultDim = resultTy.getShape()[dim];
    if (ShapedType::isDynamic(sourceDim) ||
        ShapedType::isDynamic(resultDim) || resultDim <= 0 || offset < 0 ||
        offset + resultDim > sourceDim || offset % resultDim != 0)
      return false;
  }
  return true;
}

bool hasContiguousSubviewIndices(Value value) {
  while (value) {
    if (auto require = value.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
      value = require.getSrc();
      continue;
    }
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = convert.getSrc();
      continue;
    }
    if (auto extract = value.getDefiningOp<ttg::ExtractTensorOp>()) {
      if (!isConsecutive(extract.getCtaIdx()) ||
          !isConsecutive(extract.getElemIdx()))
        return false;
      value = extract.getSource();
      continue;
    }
    if (auto extract = value.getDefiningOp<gluon_dialect::ExtractSliceOp>()) {
      if (!hasWholeTileGluonSlice(extract))
        return false;
      value = extract.getSource();
      continue;
    }
    return true;
  }
  return true;
}

bool isC500AsyncCopyContiguousSharedWrite(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto dstTy = dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!srcTy || !dstTy || srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return false;

  auto blockedEnc =
      dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  auto sharedEnc = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      sharedEncoding ? sharedEncoding : dstTy.getEncoding());
  if (!blockedEnc || !sharedEnc)
    return false;

  ArrayRef<unsigned> srcOrder = blockedEnc.getOrder();
  ArrayRef<unsigned> sharedOrder = sharedEnc.getOrder();
  if (srcOrder.size() != 2 || sharedOrder.size() != 2 ||
      srcOrder[0] != sharedOrder[0])
    return false;

  ArrayRef<unsigned> sizePerThread = blockedEnc.getSizePerThread();
  if (sizePerThread.size() != 2)
    return false;
  unsigned contiguousDim = srcOrder[0];
  unsigned elemsPerThread = sizePerThread[contiguousDim];
  if (elemsPerThread == 0)
    return false;

  unsigned elemBits = getTensorElementOrPointeeBitWidth(srcTy);
  if (elemBits == 0)
    return false;
  unsigned copyBytes = elemsPerThread * std::max<unsigned>(1, elemBits / 8);
  if (copyBytes > 16)
    return false;

  if (auto contiguity = getContiguityHint(copyOp.getSrc(), contiguousDim))
    if (*contiguity < elemsPerThread)
      return false;

  if (!hasContiguousSubviewIndices(copyOp.getSrc()))
    return false;

  if (sharedEnc.getMaxPhase() == 1)
    return true;
  unsigned sharedVec = sharedEnc.getVec();
  if (sharedVec == 0)
    return false;
  return elemsPerThread <= sharedVec;
}

LogicalResult verifyC500AsyncCopyContiguousSharedWrite(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  if (isC500AsyncCopyContiguousSharedWrite(copyOp, sharedEncoding))
    return success();
  return copyOp.emitError()
         << "C500 async_copy requires rank-2 blocked source and rank-2 "
            "swizzled shared destination with matching contiguous order, "
            "<=16-byte contiguous per-thread write, sufficient contiguity, "
            "and contiguous extract_tensor/subview indices";
}

} // namespace mlir::triton::gpu::metax::gluon::c500
