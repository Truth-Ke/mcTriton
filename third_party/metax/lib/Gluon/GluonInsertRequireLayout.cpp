#include "Gluon/Analysis/GluonLayoutPropagation.h"
#include "Gluon/GluonC500AsyncCopyLayout.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/Passes.h"
#include "TritonMETAXGPUTransforms/MACACommon.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "metax-gluon-insert-require-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;
namespace gluon_layout = ::mlir::triton::gpu::metax::gluon;
namespace c500 = ::mlir::triton::gpu::metax::gluon::c500;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONINSERTREQUIRELAYOUT
#include "Gluon/Passes.h.inc"

namespace {

static Type cloneTypeWithEncoding(Type type, Attribute encoding) {
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

static std::optional<SmallVector<unsigned, 2>>
getAsyncCopySourceOrderForMemDesc(Value memdesc);

static Attribute computeSharedEncFromDotEnc(ttg::DotOperandEncodingAttr dotEnc,
                                            ttg::LocalLoadOp localLoadOp);

static Value getExtractSource(Value value) {
  if (auto extract = value.getDefiningOp<gluon_dialect::ExtractSliceOp>())
    return extract.getSource();
  if (auto extract = value.getDefiningOp<ttg::ExtractTensorOp>())
    return extract.getSource();
  return {};
}

static Value getLoopCarriedInitValue(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return {};
  auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp)
    return {};
  int index = arg.getArgNumber() - forOp.getNumInductionVars();
  if (index < 0 || index >= static_cast<int>(forOp.getInitArgs().size()))
    return {};
  return forOp.getInitArgs()[index];
}

static std::optional<SmallVector<int64_t, 2>> getRank2TensorShape(Value value) {
  auto ty = dyn_cast<RankedTensorType>(value.getType());
  if (!ty || ty.getRank() != 2)
    return std::nullopt;
  return SmallVector<int64_t, 2>(ty.getShape().begin(), ty.getShape().end());
}

static std::optional<SmallVector<int64_t, 2>>
getLogicalMmaShapeFromAccumulator(Value value, DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return std::nullopt;

  if (Value source = getExtractSource(value))
    if (auto shape = getRank2TensorShape(source))
      return shape;

  if (auto dot = value.getDefiningOp<tt::DotOp>())
    if (auto shape = getLogicalMmaShapeFromAccumulator(dot.getC(), visited))
      return shape;

  for (Operation *user : value.getUsers()) {
    auto insert = dyn_cast<gluon_dialect::InsertSliceOp>(user);
    if (!insert || insert.getUpdate() != value)
      continue;
    if (auto shape = getRank2TensorShape(insert.getBase()))
      return shape;
    if (auto shape = getRank2TensorShape(insert.getResult()))
      return shape;
  }
  return std::nullopt;
}

static SmallVector<int64_t, 2> getLogicalMmaShape(tt::DotOp dotOp,
                                                  ArrayRef<int64_t> fallback) {
  DenseSet<Value> visited;
  if (auto shape = getLogicalMmaShapeFromAccumulator(dotOp.getC(), visited))
    return *shape;
  visited.clear();
  if (auto shape = getLogicalMmaShapeFromAccumulator(dotOp.getD(), visited))
    return *shape;
  return SmallVector<int64_t, 2>(fallback.begin(), fallback.end());
}

static int64_t getLogicalK(tt::DotOp dotOp) {
  auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
  auto bTy = cast<RankedTensorType>(dotOp.getB().getType());
  int64_t logicalK = std::max<int64_t>(aTy.getShape()[1], bTy.getShape()[0]);

  if (Value aSource = getExtractSource(dotOp.getA()))
    if (auto shape = getRank2TensorShape(aSource))
      logicalK = std::max<int64_t>(logicalK, (*shape)[1]);
  if (Value bSource = getExtractSource(dotOp.getB()))
    if (auto shape = getRank2TensorShape(bSource))
      logicalK = std::max<int64_t>(logicalK, (*shape)[0]);
  return logicalK;
}

static SmallVector<unsigned, 2>
getMetaXWarpsPerTile(tt::DotOp dotOp, ArrayRef<int64_t> shape, int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && op != dotOp)
      return {static_cast<unsigned>(numWarps), 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 16};
  do {
    if (ret[0] * ret[1] >= static_cast<unsigned>(numWarps))
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0])
        ret[0] *= 2;
      else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

static Value stripDataflowViewsForOrder(Value value) {
  while (value) {
    if (Value init = getLoopCarriedInitValue(value)) {
      value = init;
      continue;
    }
    if (Value source = getExtractSource(value)) {
      value = source;
      continue;
    }
    if (auto bsm = value.getDefiningOp<ttg::BsmPermOp>()) {
      value = bsm.getSrc1();
      continue;
    }
    if (auto require = value.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
      value = require.getSrc();
      continue;
    }
    return value;
  }
  return {};
}

static ttg::LocalLoadOp getDotPathLocalLoad(Value value) {
  DenseSet<Value> visited;
  while (value && visited.insert(value).second) {
    if (auto localLoad = value.getDefiningOp<ttg::LocalLoadOp>())
      return localLoad;
    if (Value init = getLoopCarriedInitValue(value)) {
      value = init;
      continue;
    }
    if (Value source = getExtractSource(value)) {
      value = source;
      continue;
    }
    if (auto bsm = value.getDefiningOp<ttg::BsmPermOp>()) {
      value = bsm.getSrc1();
      continue;
    }
    if (auto require = value.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
      value = require.getSrc();
      continue;
    }
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = convert.getSrc();
      continue;
    }
    return {};
  }
  return {};
}

static std::optional<SmallVector<unsigned, 2>>
getConcreteRank2Order(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getRank() != 2 || !tensorTy.getEncoding() ||
      isa<gluon_dialect::AutoEncodingAttr>(tensorTy.getEncoding()))
    return std::nullopt;
  SmallVector<unsigned> order = ttg::getOrder(tensorTy);
  if (order.size() != 2)
    return std::nullopt;
  return SmallVector<unsigned, 2>{order[0], order[1]};
}

static std::optional<SmallVector<unsigned, 2>>
getMetaXEffectiveOrderForDotOperand(Value value) {
  value = stripDataflowViewsForOrder(value);
  if (!value)
    return std::nullopt;
  if (auto order = getConcreteRank2Order(value))
    return order;
  if (auto load = value.getDefiningOp<ttg::LocalLoadOp>()) {
    if (auto asyncOrder = getAsyncCopySourceOrderForMemDesc(load.getSrc()))
      return asyncOrder;
    auto srcTy = dyn_cast<ttg::MemDescType>(load.getSrc().getType());
    if (srcTy && srcTy.getRank() == 2) {
      SmallVector<unsigned> order = ttg::getOrder(srcTy);
      if (order.size() == 2)
        return SmallVector<unsigned, 2>{order[0], order[1]};
    }
  }
  return c500::chooseRank2ContiguousOrder(value);
}

static std::optional<ttg::MACAMmaEncodingAttr>
buildC500MmaEncoding(tt::DotOp dotOp, int computeCapability,
                     bool storeCoalesce) {
  auto aTy = dyn_cast<RankedTensorType>(dotOp.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dotOp.getB().getType());
  auto dTy = dyn_cast<RankedTensorType>(dotOp.getD().getType());
  if (!aTy || !bTy || !dTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      dTy.getRank() != 2)
    return std::nullopt;

  int versionMajor = getMACAMmaVersionMajor(computeCapability);
  int versionMinor = getMACAMmaVersionMinor(computeCapability);
  if (versionMajor != 2)
    return std::nullopt;

  bool enableTf32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;
  int numWarps = ttg::lookupNumWarps(dotOp);
  SmallVector<unsigned, 3> elemsPerThread =
      getDefaultElemsPerThread(aTy.getElementType(), enableTf32,
                               computeCapability);
  SmallVector<int64_t, 2> mmaShape = getLogicalMmaShape(dotOp, dTy.getShape());
  auto warpsPerTile = getMetaXWarpsPerTile(dotOp, mmaShape, numWarps);
  int64_t logicalK = getLogicalK(dotOp);
  SmallVector<int, 4> tile{static_cast<int>(mmaShape[0]),
                           static_cast<int>(mmaShape[1]),
                           static_cast<int>(logicalK), numWarps};
  auto aOrder = getMetaXEffectiveOrderForDotOperand(dotOp.getA());
  auto bOrder = getMetaXEffectiveOrderForDotOperand(dotOp.getB());
  if (!aOrder || !bOrder)
    return std::nullopt;

  int patternVersion = -1;
  bool isOpt = updateLayout(elemsPerThread, warpsPerTile, tile, patternVersion,
                            numWarps, enableTf32, aTy.getElementType(),
                            *aOrder, *bOrder, /*disablePrefetch=*/true,
                            /*chainDot=*/false, storeCoalesce,
                            computeCapability);
  if (isOpt)
    if (auto module = dotOp->getParentOfType<ModuleOp>())
      module->setAttr(
          "use.opt.maca.mma",
          IntegerAttr::get(IntegerType::get(module.getContext(), 32), 1));

  bool aLdsTrans = getIfLdsTrans(elemsPerThread, versionMajor, versionMinor,
                                 *aOrder, /*isA=*/true,
                                 aTy.getElementType());
  bool bLdsTrans = getIfLdsTrans(elemsPerThread, versionMajor, versionMinor,
                                 *bOrder, /*isA=*/false,
                                 bTy.getElementType());
  SmallVector<unsigned, 2> elementsStride{
      aLdsTrans ? ttg::getLdsTransVec(aTy.getElementType()) : 1,
      bLdsTrans ? ttg::getLdsTransVec(bTy.getElementType()) : 1};

  return ttg::MACAMmaEncodingAttr::get(
      dotOp.getContext(), versionMajor, versionMinor, warpsPerTile,
      elemsPerThread, getDefaultMACAMmaColMajor(), aLdsTrans, bLdsTrans,
      elementsStride);
}

static Attribute getMmaSubEncoding(RankedTensorType fullTy,
                                   RankedTensorType subTy,
                                   ArrayRef<int64_t> offsets,
                                   ttg::MACAMmaEncodingAttr fullEncoding) {
  auto factors = [&]() -> std::optional<SmallVector<unsigned, 2>> {
    if (fullTy.getRank() != 2 || subTy.getRank() != 2 || offsets.size() != 2)
      return std::nullopt;
    SmallVector<unsigned, 2> result;
    for (int i = 0; i < 2; ++i) {
      int64_t fullDim = fullTy.getShape()[i];
      int64_t subDim = subTy.getShape()[i];
      if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
          fullDim <= 0 || subDim <= 0 || fullDim % subDim != 0 ||
          offsets[i] < 0 || offsets[i] % subDim != 0 ||
          offsets[i] + subDim > fullDim)
        return std::nullopt;
      result.push_back(static_cast<unsigned>(fullDim / subDim));
    }
    return result;
  }();
  if (!factors)
    return fullEncoding;

  SmallVector<unsigned, 3> elems(fullEncoding.getElementsMNK());
  elems[0] = std::max<unsigned>(1, elems[0] / (*factors)[0]);
  elems[1] = std::max<unsigned>(1, elems[1] / (*factors)[1]);
  return ttg::MACAMmaEncodingAttr::get(
      fullTy.getContext(), fullEncoding.getVersionMajor(),
      fullEncoding.getVersionMinor(), fullEncoding.getWarpsPerCTA(), elems,
      fullEncoding.getColMajor(), fullEncoding.getIsATrans(),
      fullEncoding.getIsBTrans(), fullEncoding.getElementsStride());
}

static Attribute inferDotAccumulatorSubEncoding(
    tt::DotOp dotOp, ttg::MACAMmaEncodingAttr fullEncoding) {
  if (auto extract = dotOp.getC().getDefiningOp<gluon_dialect::ExtractSliceOp>()) {
    auto fullTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
    auto subTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
    if (fullTy && subTy)
      return getMmaSubEncoding(fullTy, subTy, extract.getOffsets(),
                               fullEncoding);
  }

  for (Operation *user : dotOp.getD().getUsers()) {
    auto insert = dyn_cast<gluon_dialect::InsertSliceOp>(user);
    if (!insert || insert.getUpdate() != dotOp.getD())
      continue;
    auto fullTy = dyn_cast<RankedTensorType>(insert.getBase().getType());
    auto subTy = dyn_cast<RankedTensorType>(insert.getUpdate().getType());
    if (fullTy && subTy)
      return getMmaSubEncoding(fullTy, subTy, insert.getOffsets(),
                               fullEncoding);
  }
  return fullEncoding;
}

static unsigned getStoreCoalesceElemN(ModuleOp module) {
  unsigned elemN = 1;
  module.walk([&](gluon_dialect::SetAutoLayoutOp op) {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return;
    if (auto mma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
            type.getEncoding()))
      elemN = std::max(elemN, mma.getElementsMNK()[1]);
  });
  return elemN;
}

static std::optional<ttg::BlockedEncodingAttr>
inferC500AsyncCopyExtractParentEncoding(gluon_dialect::ExtractSliceOp extract,
                                        Attribute subEncoding,
                                        unsigned storeCoalesceElemN,
                                        Operation *anchor) {
  auto subBlocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(subEncoding);
  if (!subBlocked)
    return std::nullopt;

  auto parent =
      c500::getC500GlobalBlockedEncodingLike(extract.getSource(), anchor,
                                             subBlocked);
  if (!parent || storeCoalesceElemN <= 1)
    return parent;

  ArrayRef<unsigned> order = subBlocked.getOrder();
  auto sourceTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!sourceTy || !resultTy || sourceTy.getRank() != 2 ||
      resultTy.getRank() != 2 || order.size() != 2 || order[0] != 0 ||
      order[1] != 1)
    return parent;

  ArrayRef<int64_t> sourceShape = sourceTy.getShape();
  ArrayRef<int64_t> resultShape = resultTy.getShape();
  if (sourceShape[0] != resultShape[0] ||
      sourceShape[1] <= resultShape[1])
    return parent;

  SmallVector<unsigned> sizePerThread(parent->getSizePerThread());
  sizePerThread[order[1]] = storeCoalesceElemN;
  return ttg::BlockedEncodingAttr::get(
      extract.getContext(), sizePerThread, parent->getThreadsPerWarp(),
      parent->getWarpsPerCTA(), parent->getOrder(), parent->getCTALayout());
}

static bool hasAutoEncoding(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  return tensorTy &&
         isa<gluon_dialect::AutoEncodingAttr>(tensorTy.getEncoding());
}

static LogicalResult
insertSetAutoLayoutSeed(Value value, Attribute encoding, OpBuilder &builder,
                        llvm::DenseMap<Value, Attribute> &seeds,
                        Operation *diagnosticOp) {
  if (!value || !encoding || !hasAutoEncoding(value))
    return success();

  auto [it, inserted] = seeds.try_emplace(value, encoding);
  if (!inserted) {
    if (it->second == encoding)
      return success();
    return diagnosticOp->emitError()
           << "found conflicting MetaX Gluon auto-layout seeds for value "
           << value << ": " << it->second << " vs " << encoding;
  }

  if (auto result = dyn_cast<OpResult>(value)) {
    builder.setInsertionPointAfter(result.getOwner());
  } else if (auto arg = dyn_cast<BlockArgument>(value)) {
    builder.setInsertionPointToStart(arg.getOwner());
  } else {
    return diagnosticOp->emitError()
           << "cannot place MetaX Gluon auto-layout seed for value " << value;
  }
  builder.create<gluon_dialect::SetAutoLayoutOp>(value.getLoc(), encoding,
                                                 value);
  return success();
}

static LogicalResult seedAsyncCopyExtractParentLayout(
    Value value, Attribute subEncoding, unsigned storeCoalesceElemN,
    OpBuilder &builder, llvm::DenseMap<Value, Attribute> &autoLayoutSeeds,
    Operation *diagnosticOp) {
  auto extract = value.getDefiningOp<gluon_dialect::ExtractSliceOp>();
  if (!extract)
    return success();

  auto parentEncoding = inferC500AsyncCopyExtractParentEncoding(
      extract, subEncoding, storeCoalesceElemN, diagnosticOp);
  if (!parentEncoding)
    return success();

  return insertSetAutoLayoutSeed(extract.getSource(), *parentEncoding, builder,
                                 autoLayoutSeeds, diagnosticOp);
}

static void requireTensorOperand(Operation *owner, unsigned operandIndex,
                                 Attribute encoding, OpBuilder &builder) {
  Value operand = owner->getOperand(operandIndex);
  auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorTy || !encoding || tensorTy.getEncoding() == encoding)
    return;

  builder.setInsertionPoint(owner);
  auto requiredTy = cast<RankedTensorType>(
      cloneTypeWithEncoding(tensorTy, encoding));
  auto require = builder.create<gluon_dialect::RequireLayoutOp>(
      owner->getLoc(), requiredTy, operand);
  owner->setOperand(operandIndex, require.getResult());
}

static LogicalResult
checkMemDescRootEncoding(Value memdesc, Attribute encoding,
                         Operation *diagnosticOp,
                         llvm::DenseMap<Value, Attribute> &rootEncodings) {
  Value root = c500::getMemDescRoot(memdesc);
  if (!root)
    return success();

  auto [it, inserted] = rootEncodings.try_emplace(root, encoding);
  if (inserted || it->second == encoding)
    return success();

  return diagnosticOp->emitError()
         << "conflicting shared encodings for the same memdesc family: "
         << it->second << " vs " << encoding;
}

static void requireMemDescOperand(Operation *owner, unsigned operandIndex,
                                  Attribute encoding, OpBuilder &builder) {
  Value operand = owner->getOperand(operandIndex);
  auto memdescTy = dyn_cast<ttg::MemDescType>(operand.getType());
  if (!memdescTy || !encoding || memdescTy.getEncoding() == encoding)
    return;
  if (operand.getDefiningOp<gluon_dialect::RequireLayoutOp>())
    return;

  builder.setInsertionPoint(owner);
  auto requiredTy =
      cast<ttg::MemDescType>(cloneTypeWithEncoding(memdescTy, encoding));
  auto require = builder.create<gluon_dialect::RequireLayoutOp>(
      owner->getLoc(), requiredTy, operand);
  owner->setOperand(operandIndex, require.getResult());
}

static LogicalResult
requireDotLocalLoadMemDesc(ttg::LocalLoadOp localLoadOp,
                           ttg::DotOperandEncodingAttr dotEnc,
                           OpBuilder &builder,
                           llvm::DenseMap<Value, Attribute> &rootEncodings) {
  Attribute sharedEnc = computeSharedEncFromDotEnc(dotEnc, localLoadOp);
  if (failed(checkMemDescRootEncoding(localLoadOp.getSrc(), sharedEnc,
                                      localLoadOp, rootEncodings)))
    return failure();

  if (auto require =
          localLoadOp.getSrc().getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
    auto requireTy = cast<ttg::MemDescType>(require.getType());
    if (requireTy.getEncoding() != sharedEnc)
      return localLoadOp.emitError()
             << "conflicting pre-existing local_load memdesc require_layout: "
             << requireTy.getEncoding() << " vs " << sharedEnc;
    return success();
  }

  requireMemDescOperand(localLoadOp, /*operandIndex=*/0, sharedEnc, builder);
  return success();
}

static LogicalResult materializeMetaXDotConstraints(ModuleOp module,
                                                    int computeCapability,
                                                    bool storeCoalesce,
                                                    OpBuilder &builder) {
  llvm::DenseMap<Value, Attribute> autoLayoutSeeds;
  llvm::DenseMap<Value, Attribute> memDescRootEncodings;

  WalkResult result = module.walk([&](tt::DotOp dotOp) {
    Value originalA = dotOp.getA();
    Value originalB = dotOp.getB();

    auto fullMma =
        buildC500MmaEncoding(dotOp, computeCapability, storeCoalesce);
    if (!fullMma)
      return WalkResult::advance();

    Attribute dotMmaEnc = inferDotAccumulatorSubEncoding(dotOp, *fullMma);
    auto dotMma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(dotMmaEnc);
    if (!dotMma) {
      dotOp.emitError() << "MetaX Gluon dot accumulator layout must be a "
                           "MACA MMA encoding";
      return WalkResult::interrupt();
    }

    auto aTy = dyn_cast<RankedTensorType>(originalA.getType());
    auto bTy = dyn_cast<RankedTensorType>(originalB.getType());
    if (!aTy || !bTy) {
      dotOp.emitError() << "MetaX Gluon dot operands must be ranked tensors";
      return WalkResult::interrupt();
    }

    auto aEnc = ttg::DotOperandEncodingAttr::get(
        dotOp.getContext(), /*opIdx=*/0, dotMma, aTy.getElementType());
    auto bEnc = ttg::DotOperandEncodingAttr::get(
        dotOp.getContext(), /*opIdx=*/1, dotMma, bTy.getElementType());

    if (failed(insertSetAutoLayoutSeed(dotOp.getC(), dotMma, builder,
                                       autoLayoutSeeds, dotOp)) ||
        failed(insertSetAutoLayoutSeed(dotOp.getD(), dotMma, builder,
                                       autoLayoutSeeds, dotOp)))
      return WalkResult::interrupt();

    if (auto extract =
            dotOp.getC().getDefiningOp<gluon_dialect::ExtractSliceOp>()) {
      if (failed(insertSetAutoLayoutSeed(extract.getSource(), *fullMma,
                                         builder, autoLayoutSeeds, dotOp)))
        return WalkResult::interrupt();
    }

    for (Operation *user : dotOp.getD().getUsers()) {
      auto insert = dyn_cast<gluon_dialect::InsertSliceOp>(user);
      if (!insert || insert.getUpdate() != dotOp.getD())
        continue;
      if (failed(insertSetAutoLayoutSeed(insert.getBase(), *fullMma, builder,
                                         autoLayoutSeeds, dotOp)) ||
          failed(insertSetAutoLayoutSeed(insert.getResult(), *fullMma, builder,
                                         autoLayoutSeeds, dotOp)))
        return WalkResult::interrupt();
    }

    if (auto localLoad = getDotPathLocalLoad(originalA))
      if (failed(requireDotLocalLoadMemDesc(localLoad, aEnc, builder,
                                            memDescRootEncodings)))
        return WalkResult::interrupt();
    if (auto localLoad = getDotPathLocalLoad(originalB))
      if (failed(requireDotLocalLoadMemDesc(localLoad, bEnc, builder,
                                            memDescRootEncodings)))
        return WalkResult::interrupt();

    requireTensorOperand(dotOp, /*operandIndex=*/0, aEnc, builder);
    requireTensorOperand(dotOp, /*operandIndex=*/1, bEnc, builder);
    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult materializeC500GlobalMemoryConstraints(ModuleOp module,
                                                            OpBuilder &builder) {
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(module);
  llvm::DenseMap<Value, Attribute> autoLayoutSeeds;

  WalkResult loadResult = module.walk([&](tt::LoadOp loadOp) {
    auto ptrTy = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
    if (!ptrTy || ptrTy.getRank() != 2)
      return WalkResult::advance();

    auto ptrOrder =
        c500::chooseRank2ContiguousOrder(loadOp.getPtr(), &axisInfoAnalysis);
    auto ptrEncoding =
        c500::getC500GlobalBlockedEncoding(loadOp.getPtr(), loadOp, ptrOrder);
    if (!ptrEncoding) {
      loadOp.emitError()
          << "C500 load pointer currently requires a rank-2 tensor ptr";
      return WalkResult::interrupt();
    }

    if (failed(insertSetAutoLayoutSeed(loadOp.getPtr(), *ptrEncoding,
                                       builder, autoLayoutSeeds, loadOp)))
      return WalkResult::interrupt();
    if (failed(insertSetAutoLayoutSeed(loadOp.getResult(), *ptrEncoding,
                                       builder, autoLayoutSeeds, loadOp)))
      return WalkResult::interrupt();
    if (loadOp.getMask())
      if (failed(insertSetAutoLayoutSeed(loadOp.getMask(), *ptrEncoding,
                                         builder, autoLayoutSeeds, loadOp)))
        return WalkResult::interrupt();
    if (loadOp.getOther())
      if (failed(insertSetAutoLayoutSeed(loadOp.getOther(), *ptrEncoding,
                                         builder, autoLayoutSeeds, loadOp)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (loadResult.wasInterrupted())
    return failure();

  WalkResult storeResult = module.walk([&](tt::StoreOp storeOp) {
    auto ptrTy = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());
    if (!ptrTy || ptrTy.getRank() != 2)
      return WalkResult::advance();

    auto ptrOrder =
        c500::chooseRank2ContiguousOrder(storeOp.getPtr(), &axisInfoAnalysis);
    auto ptrEncoding =
        c500::getC500GlobalBlockedEncoding(storeOp.getPtr(), storeOp,
                                           ptrOrder);
    if (!ptrEncoding) {
      storeOp.emitError()
          << "C500 store pointer currently requires a rank-2 tensor ptr";
      return WalkResult::interrupt();
    }

    if (failed(insertSetAutoLayoutSeed(storeOp.getPtr(), *ptrEncoding,
                                       builder, autoLayoutSeeds, storeOp)))
      return WalkResult::interrupt();
    if (storeOp.getMask())
      if (failed(insertSetAutoLayoutSeed(storeOp.getMask(), *ptrEncoding,
                                         builder, autoLayoutSeeds, storeOp)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return storeResult.wasInterrupted() ? failure() : success();
}

class DotRewriteState {
public:
  enum class Kind { Uninitialized, Required, Conflict, Illegal };

  DotRewriteState() = default;
  explicit DotRewriteState(Attribute enc) : kind(Kind::Required), encoding(enc) {}

  static DotRewriteState getConflict() {
    DotRewriteState state;
    state.kind = Kind::Conflict;
    return state;
  }
  static DotRewriteState getIllegal() {
    DotRewriteState state;
    state.kind = Kind::Illegal;
    return state;
  }

  bool operator==(const DotRewriteState &rhs) const {
    return kind == rhs.kind && encoding == rhs.encoding;
  }
  bool isUninitialized() const { return kind == Kind::Uninitialized; }
  bool isRequired() const { return kind == Kind::Required; }
  bool isConflict() const { return kind == Kind::Conflict; }
  bool isIllegal() const { return kind == Kind::Illegal; }
  Attribute getEncoding() const {
    assert(isRequired());
    return *encoding;
  }

  void print(llvm::raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    if (isConflict()) {
      os << "<conflict>";
      return;
    }
    if (isIllegal()) {
      os << "<illegal>";
      return;
    }
    getEncoding().print(os);
  }

  static DotRewriteState meet(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    if (lhs.isIllegal() || rhs.isIllegal())
      return getIllegal();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    if (lhs.isConflict() || rhs.isConflict())
      return getConflict();
    return getConflict();
  }
  static DotRewriteState join(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    return meet(lhs, rhs);
  }

private:
  Kind kind = Kind::Uninitialized;
  std::optional<Attribute> encoding;
};

class DotRewriteLattice : public dataflow::Lattice<DotRewriteState> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DotRewriteLattice)
  using Lattice::Lattice;
};

static bool isTrackedDotValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

static bool isTransparentDotUser(Operation *op, unsigned operandIndex) {
  if (auto dotOp = dyn_cast<tt::DotOp>(op))
    return operandIndex < 2 && operandIndex < dotOp->getNumOperands();
  return isa<ttg::ConvertLayoutOp>(op) ||
         gluon_layout::isTransparentLayoutCarrierOp(op);
}

class DotRewriteBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<DotRewriteLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void initializeEquivalentLatticeAnchor(Operation *top) override {
    top->walk([&](ttg::ConvertLayoutOp cvt) {
      if (!isTrackedDotValue(cvt.getSrc()) ||
          !isTrackedDotValue(cvt.getResult()))
        return;
      unionLatticeAnchors<DotRewriteLattice>(cvt.getSrc(), cvt.getResult());
    });
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotRewriteLattice *> operands,
                 ArrayRef<const DotRewriteLattice *> results) override {
    if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      for (unsigned i = 0; i < 2; ++i) {
        auto type = cast<RankedTensorType>(dotOp.getOperand(i).getType());
        if (auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(type.getEncoding())) {
          ChangeResult changed = operands[i]->meet(DotRewriteState(dotEnc));
          propagateIfChanged(operands[i], changed);
        }
      }
      return success();
    }

    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isTrackedDotValue(operand))
        continue;
      if (isTransparentDotUser(op, index))
        continue;
      DotRewriteState state = operands[index]->getValue();
      if (state.isUninitialized())
        continue;
      if (remarkedUnsupportedUsers.insert(op).second)
        op->emitRemark()
            << "MetaX Gluon dot layout requirement cannot be absorbed through "
               "operand "
            << index
            << "; propagate-layout will materialize a tensor convert if needed";
      ChangeResult changed =
          operands[index]->meet(DotRewriteState::getIllegal());
      propagateIfChanged(operands[index], changed);
    }
    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {
    if (!isTrackedDotValue(operand.get()))
      return;
    if (gluon_layout::isTransparentLayoutCarrierOp(operand.getOwner()))
      return;
    poison(operand);
  }
  void visitCallOperand(OpOperand &operand) override { poison(operand); }
  void setToExitState(DotRewriteLattice *) override {}

private:
  DenseSet<Operation *> remarkedUnsupportedUsers;

  void poison(OpOperand &operand) {
    auto *lattice = getLatticeElement(operand.get());
    if (lattice->getValue().isUninitialized())
      return;
    ChangeResult changed = lattice->meet(DotRewriteState::getIllegal());
    propagateIfChanged(lattice, changed);
  }
};

using DotConsumerInfo = c500::DotConsumerInfo;

class DotConsumerState {
public:
  enum class Kind { Uninitialized, Required, Conflict };

  DotConsumerState() = default;
  explicit DotConsumerState(DotConsumerInfo info)
      : kind(Kind::Required), info(std::move(info)) {}

  static DotConsumerState getConflict() {
    DotConsumerState state;
    state.kind = Kind::Conflict;
    return state;
  }

  bool operator==(const DotConsumerState &rhs) const {
    return kind == rhs.kind && info == rhs.info;
  }
  bool isUninitialized() const { return kind == Kind::Uninitialized; }
  bool isRequired() const { return kind == Kind::Required; }
  bool isConflict() const { return kind == Kind::Conflict; }
  DotConsumerInfo getInfo() const {
    assert(isRequired());
    return info;
  }

  void print(llvm::raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    if (isConflict()) {
      os << "<conflict>";
      return;
    }
    os << "<required dot consumer ";
    os << info.dotEnc;
    os << ">";
  }

  static DotConsumerState meet(const DotConsumerState &lhs,
                               const DotConsumerState &rhs) {
    if (lhs.isConflict() || rhs.isConflict())
      return getConflict();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    return getConflict();
  }
  static DotConsumerState join(const DotConsumerState &lhs,
                               const DotConsumerState &rhs) {
    return meet(lhs, rhs);
  }

private:
  Kind kind = Kind::Uninitialized;
  DotConsumerInfo info{};
};

class DotConsumerLattice : public dataflow::Lattice<DotConsumerState> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DotConsumerLattice)
  using Lattice::Lattice;
};

static std::optional<SmallVector<unsigned>>
getUserSharedOrder(ttg::LocalLoadOp localLoadOp) {
  auto loadMemDesc = localLoadOp->getOperand(0);
  if (auto srcType = dyn_cast<ttg::MemDescType>(loadMemDesc.getType())) {
    if (auto srcEnc =
            dyn_cast_or_null<ttg::SharedEncodingTrait>(srcType.getEncoding()))
      return ttg::getOrder(srcEnc, srcType.getShape());
  }
  return std::nullopt;
}

class DotConsumerBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<DotConsumerLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotConsumerLattice *> operands,
                 ArrayRef<const DotConsumerLattice *> results) override {
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      auto resultTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
      if (!resultTy)
        return success();
      auto dotEnc =
          dyn_cast_or_null<ttg::DotOperandEncodingAttr>(resultTy.getEncoding());
      if (!dotEnc || operands.empty())
        return success();

      DotConsumerState seed(DotConsumerInfo{
          dotEnc, SmallVector<int64_t>(resultTy.getShape()),
          resultTy.getElementType(), getUserSharedOrder(loadOp)});
      ChangeResult changed = operands[0]->meet(seed);
      propagateIfChanged(operands[0], changed);
      return success();
    }

    if (isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
            ttg::MemDescSubsliceOp, ttg::MemDescTransOp,
            ttg::MemDescReshapeOp, gluon_dialect::RequireLayoutOp>(op)) {
      for (const DotConsumerLattice *resultLattice : results) {
        for (auto [i, operandLattice] : llvm::enumerate(operands)) {
          if (!isa<ttg::MemDescType>(op->getOpOperand(i).get().getType()))
            continue;
          ChangeResult changed = operandLattice->meet(resultLattice->getValue());
          propagateIfChanged(operandLattice, changed);
        }
      }
    }
    return success();
  }

  void visitBranchOperand(OpOperand &) override {}
  void visitCallOperand(OpOperand &) override {}
  void setToExitState(DotConsumerLattice *) override {}
};

static std::optional<DotConsumerInfo> findDotConsumer(Value buffer,
                                                      DataFlowSolver &solver) {
  Value root = c500::getMemDescRoot(buffer);
  if (!root)
    return std::nullopt;
  auto *lattice = solver.lookupState<DotConsumerLattice>(root);
  if (!lattice)
    return std::nullopt;
  const DotConsumerState &state = lattice->getValue();
  if (!state.isRequired())
    return std::nullopt;
  return state.getInfo();
}

static std::optional<SmallVector<unsigned, 2>>
getAsyncCopySourceOrderForMemDesc(Value memdesc) {
  Value root = c500::getMemDescRoot(memdesc);
  if (!root)
    return std::nullopt;
  llvm::SetVector<Value> aliases;
  c500::collectMemDescAliasValues(root, aliases);

  std::optional<SmallVector<unsigned, 2>> order;
  for (Value alias : aliases) {
    for (Operation *user : alias.getUsers()) {
      auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(user);
      if (!copyOp || copyOp->getOperand(1) != alias)
        continue;
      auto candidate = c500::chooseRank2ContiguousOrder(copyOp.getSrc());
      if (order && *order != candidate)
        return std::nullopt;
      order = candidate;
    }
  }
  return order;
}

static LogicalResult
findExistingMemDescRequireEncoding(Value memdesc, Operation *diagnosticOp,
                                   std::optional<Attribute> &encoding) {
  Value root = c500::getMemDescRoot(memdesc);
  if (!root)
    return success();

  llvm::SetVector<Value> aliases;
  c500::collectMemDescAliasValues(root, aliases);
  for (Value alias : aliases) {
    for (Operation *user : alias.getUsers()) {
      auto require = dyn_cast<gluon_dialect::RequireLayoutOp>(user);
      if (!require || require.getSrc() != alias ||
          !isa<ttg::MemDescType>(require.getType()))
        continue;

      Attribute candidate =
          cast<ttg::MemDescType>(require.getType()).getEncoding();
      if (!candidate)
        continue;
      if (encoding && *encoding != candidate)
        return diagnosticOp->emitError()
               << "conflicting pre-existing memdesc require_layout encodings "
                  "for the same buffer family: "
               << *encoding << " vs " << candidate;
      encoding = candidate;
    }
  }
  return success();
}

static Attribute computeSharedEncFromDotEnc(ttg::DotOperandEncodingAttr dotEnc,
                                            ttg::LocalLoadOp localLoadOp) {
  auto resultType = cast<RankedTensorType>(localLoadOp.getType());
  SmallVector<unsigned> order;
  if (auto asyncOrder =
          getAsyncCopySourceOrderForMemDesc(localLoadOp->getOperand(0)))
    order.assign(asyncOrder->begin(), asyncOrder->end());
  else if (auto userOrder = getUserSharedOrder(localLoadOp))
    order = *userOrder;
  else
    order = ttg::getOrderForMemory(resultType);

  auto ctaLayout = ttg::getCTALayout(dotEnc);
  return ttg::SwizzledSharedEncodingAttr::get(
      localLoadOp->getContext(), dotEnc, resultType.getShape(), order,
      ctaLayout, resultType.getElementType(), /*needTrans=*/false);
}

static void materializeTensorRequireLayout(tt::DotOp dotOp,
                                           unsigned operandIndex,
                                           OpBuilder &builder) {
  Value operand = dotOp.getOperand(operandIndex);
  auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!cvt)
    return;
  auto dstType = dyn_cast<RankedTensorType>(cvt.getType());
  if (!dstType ||
      !gluon_layout::isSupportedDotConstraintEncoding(dstType.getEncoding()))
    return;

  builder.setInsertionPoint(cvt);
  auto require = builder.create<gluon_dialect::RequireLayoutOp>(
      cvt.getLoc(), cvt.getType(), cvt.getSrc());
  dotOp->setOperand(operandIndex, require.getResult());
  if (cvt.getResult().use_empty())
    cvt.erase();
}

static void materializeDotUserTensorConstraints(ModuleOp module,
                                                OpBuilder &builder) {
  module.walk([&](tt::DotOp dotOp) {
    materializeTensorRequireLayout(dotOp, 0, builder);
    materializeTensorRequireLayout(dotOp, 1, builder);
  });
}

static LogicalResult anchorC500AsyncCopyRequireLayout(
    ttg::AsyncCopyGlobalToLocalOp copyOp, OpBuilder &builder,
    DataFlowSolver &solver, llvm::DenseMap<Value, Attribute> &rootEncodings) {
  Value dstMemDesc = copyOp->getOperand(1);
  auto dstTy = dyn_cast<ttg::MemDescType>(dstMemDesc.getType());
  if (!dstTy)
    return copyOp.emitError()
           << "C500 async_copy destination must be a memdesc";

  std::optional<Attribute> existingSharedEncoding;
  if (failed(findExistingMemDescRequireEncoding(dstMemDesc, copyOp,
                                                existingSharedEncoding)))
    return failure();

  Attribute sharedEncoding;
  if (existingSharedEncoding) {
    sharedEncoding = *existingSharedEncoding;
  } else {
    FailureOr<Attribute> chosen = c500::chooseAsyncCopySharedEncoding(
        copyOp, dstMemDesc, findDotConsumer(dstMemDesc, solver));
    if (failed(chosen))
      return failure();
    sharedEncoding = *chosen;
  }

  if (failed(c500::verifyC500AsyncCopyContiguousSharedWrite(copyOp,
                                                            sharedEncoding)))
    return failure();

  Value root = c500::getMemDescRoot(dstMemDesc);
  auto [it, inserted] = rootEncodings.try_emplace(root, sharedEncoding);
  if (!inserted && it->second != sharedEncoding)
    return copyOp.emitError()
           << "conflicting C500 async_copy shared encodings for the same "
              "destination buffer family: "
           << it->second << " vs " << sharedEncoding;

  if (auto require =
          dstMemDesc.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
    auto requireTy = cast<ttg::MemDescType>(require.getType());
    if (requireTy.getEncoding() != sharedEncoding)
      return copyOp.emitError()
             << "conflicting pre-existing memdesc require_layout for "
                "C500 async_copy destination: "
             << requireTy.getEncoding() << " vs " << sharedEncoding;
    return success();
  }

  requireMemDescOperand(copyOp, /*operandIndex=*/1, sharedEncoding, builder);
  return success();
}

static LogicalResult materializeC500AsyncCopyConstraints(ModuleOp module,
                                                         OpBuilder &builder,
                                                         DataFlowSolver &solver,
                                                         unsigned storeCoalesceElemN) {
  llvm::DenseMap<Value, Attribute> rootEncodings;
  llvm::DenseMap<Value, Attribute> autoLayoutSeeds;
  SmallVector<ttg::AsyncCopyGlobalToLocalOp> copyOps;
  module.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
    copyOps.push_back(copyOp);
  });

  for (ttg::AsyncCopyGlobalToLocalOp copyOp : copyOps) {
    auto srcOrder = c500::chooseRank2ContiguousOrder(copyOp.getSrc());
    auto srcEncoding =
        c500::getC500GlobalBlockedEncoding(copyOp.getSrc(), copyOp, srcOrder);
    if (!srcEncoding) {
      copyOp.emitError()
          << "C500 async_copy source currently requires rank-2 tensor src";
      return failure();
    }

    if (failed(seedAsyncCopyExtractParentLayout(
            copyOp.getSrc(), *srcEncoding, storeCoalesceElemN, builder,
            autoLayoutSeeds, copyOp)))
      return failure();
    if (copyOp.getMask() &&
        failed(seedAsyncCopyExtractParentLayout(
            copyOp.getMask(), *srcEncoding, storeCoalesceElemN, builder,
            autoLayoutSeeds, copyOp)))
      return failure();
    if (copyOp.getOther() &&
        failed(seedAsyncCopyExtractParentLayout(
            copyOp.getOther(), *srcEncoding, storeCoalesceElemN, builder,
            autoLayoutSeeds, copyOp)))
      return failure();

    requireTensorOperand(copyOp, /*operandIndex=*/0, *srcEncoding, builder);
    if (copyOp.getMask())
      requireTensorOperand(copyOp, /*operandIndex=*/2, *srcEncoding, builder);
    if (copyOp.getOther())
      requireTensorOperand(copyOp, /*operandIndex=*/3, *srcEncoding, builder);

    if (failed(anchorC500AsyncCopyRequireLayout(copyOp, builder, solver,
                                                rootEncodings)))
      return failure();
  }
  return success();
}

static LogicalResult insertRequireLayout(ModuleOp module, int computeCapability,
                                         bool storeCoalesce) {
  OpBuilder builder(module.getContext());
  if (failed(materializeMetaXDotConstraints(module, computeCapability,
                                            storeCoalesce, builder)))
    return failure();
  if (failed(materializeC500GlobalMemoryConstraints(module, builder)))
    return failure();

  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<DotRewriteBackward>(symbolTable);
  solver.load<DotConsumerBackward>(symbolTable);
  if (failed(solver.initializeAndRun(module)))
    return failure();

  module.walk([&](ttg::LocalLoadOp localLoadOp) {
    if (c500::isFedByC500AsyncCopyProducer(localLoadOp->getOperand(0)))
      return;
    auto *lattice =
        solver.lookupState<DotRewriteLattice>(localLoadOp.getResult());
    if (!lattice)
      return;
    const DotRewriteState &state = lattice->getValue();
    if (!state.isRequired())
      return;
    auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(state.getEncoding());
    if (!dotEnc)
      return;
    Attribute sharedEnc = computeSharedEncFromDotEnc(dotEnc, localLoadOp);
    requireMemDescOperand(localLoadOp, /*operandIndex=*/0, sharedEnc, builder);
  });

  materializeDotUserTensorConstraints(module, builder);
  unsigned storeCoalesceElemN = getStoreCoalesceElemN(module);
  return materializeC500AsyncCopyConstraints(module, builder, solver,
                                            storeCoalesceElemN);
}

class TritonMETAXGPUGluonInsertRequireLayoutPass
    : public impl::TritonMETAXGPUGluonInsertRequireLayoutBase<
          TritonMETAXGPUGluonInsertRequireLayoutPass> {
public:
  TritonMETAXGPUGluonInsertRequireLayoutPass() = default;
  explicit TritonMETAXGPUGluonInsertRequireLayoutPass(int computeCapability,
                                                      bool storeCoalesce) {
    this->computeCapability = computeCapability;
    this->storeCoalesce = storeCoalesce;
  }

  void runOnOperation() override {
    if (failed(insertRequireLayout(getOperation(), computeCapability,
                                   storeCoalesce)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTritonMETAXGPUGluonInsertRequireLayoutPass(
    int computeCapability, bool storeCoalesce) {
  return std::make_unique<TritonMETAXGPUGluonInsertRequireLayoutPass>(
      computeCapability, storeCoalesce);
}

} // namespace mlir
