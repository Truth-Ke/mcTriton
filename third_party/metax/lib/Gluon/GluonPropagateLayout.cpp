#include "Gluon/Analysis/GluonLayoutPropagation.h"
#include "Gluon/GluonC500AsyncCopyLayout.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/Passes.h"
#include "TritonMETAXGPUTransforms/MACACommon.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "metax-gluon-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;
namespace gluon_layout = ::mlir::triton::gpu::metax::gluon;
namespace c500 = ::mlir::triton::gpu::metax::gluon::c500;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONPROPAGATELAYOUT
#include "Gluon/Passes.h.inc"

namespace {

static bool isAutoEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy &&
         isa<gluon_dialect::AutoEncodingAttr>(tensorTy.getEncoding());
}

static bool hasAutoEncoding(Value value) {
  return isAutoEncodingTensorType(value.getType());
}

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

static Attribute getTypeEncoding(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return unwrapNoVerifyEncoding(tensorTy.getEncoding());
  if (auto memdescTy = dyn_cast<ttg::MemDescType>(type))
    return unwrapNoVerifyEncoding(memdescTy.getEncoding());
  return {};
}

static void setValueTypeWithEncoding(Value value, Attribute encoding) {
  if (!value || !encoding)
    return;
  Type newType = cloneTypeWithEncoding(value.getType(), encoding);
  if (newType != value.getType())
    value.setType(newType);
}

static void collectLayoutSeeds(
    tt::FuncOp func, SmallVectorImpl<std::pair<Value, Attribute>> &seeds) {
  // set_auto_layout is only a seed for AutoEncoding value solving. Hard
  // hardware boundaries, such as dot-fed local_load memdesc and async_copy
  // destination memdesc, are expressed with require_layout contracts.
  func.walk([&](gluon_dialect::SetAutoLayoutOp op) {
    seeds.push_back({op.getSrc(), getTypeEncoding(op.getType())});
  });
  func.walk([&](gluon_dialect::RequireLayoutOp op) {
    if (isa<RankedTensorType>(op.getSrc().getType()))
      seeds.push_back({op.getSrc(), getTypeEncoding(op.getType())});
  });
}

static ttg::MemDescType getNewMemDescType(ttg::MemDescType origType,
                                          Attribute encoding) {
  return ttg::MemDescType::get(origType.getShape(), origType.getElementType(),
                               encoding, origType.getMemorySpace(),
                               origType.getMutableMemory(),
                               origType.getAllocShape());
}

static FailureOr<const gluon_layout::LayoutEncodingLattice *>
lookupMemDescLatticeOrEmitError(Value value, DataFlowSolver &solver,
                                Operation *diagnosticOp) {
  auto *lattice = solver.lookupState<gluon_layout::LayoutEncodingLattice>(value);
  if (lattice)
    return lattice;
  diagnosticOp->emitError() << "expected memdesc layout lattice for value "
                            << value;
  return failure();
}

static LogicalResult rewriteMemDescValueFromLattice(
    Value value, DataFlowSolver &solver, Operation *diagnosticOp) {
  auto origType = dyn_cast<ttg::MemDescType>(value.getType());
  if (!origType)
    return success();

  FailureOr<const gluon_layout::LayoutEncodingLattice *> lattice =
      lookupMemDescLatticeOrEmitError(value, solver, diagnosticOp);
  if (failed(lattice))
    return failure();

  gluon_layout::LayoutEncoding layout = (*lattice)->getValue();
  if (layout.isUninitialized() || layout.isUnknown())
    return success();

  auto newType = getNewMemDescType(origType, layout.getLayoutEncoding());
  if (newType != origType) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (auto func = dyn_cast_or_null<tt::FuncOp>(
              arg.getOwner()->getParentOp())) {
        SmallVector<Type> inputs(func.getFunctionType().getInputs());
        if (arg.getArgNumber() < inputs.size()) {
          inputs[arg.getArgNumber()] = newType;
          func.setFunctionType(FunctionType::get(
              func.getContext(), inputs, func.getFunctionType().getResults()));
        }
      }
    }
    value.setType(newType);
  }
  return success();
}

static RankedTensorType getNewTensorType(RankedTensorType origType,
                                         Attribute encoding) {
  return RankedTensorType::get(origType.getShape(), origType.getElementType(),
                               encoding);
}

static bool isRetaggableTensorProducerValue(Value value) {
  if (!isa<RankedTensorType>(value.getType()))
    return false;
  Operation *definingOp = value.getDefiningOp();
  return isa_and_nonnull<ttg::LocalLoadOp, RegionBranchOpInterface>(definingOp);
}

static Type getTensorCandidateType(
    Value value, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedTensorValues) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  if (blockedTensorValues.contains(value))
    return tensorType;
  auto *lattice = solver.lookupState<gluon_layout::TensorLayoutLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized() ||
      lattice->getValue().isUnknown())
    return tensorType;
  return getNewTensorType(tensorType, lattice->getValue().getLayoutEncoding());
}

static void
rewriteTensorValueFromLattice(Value value, DataFlowSolver &solver,
                              const llvm::DenseSet<Value> &blockedTensorValues) {
  if (!isRetaggableTensorProducerValue(value))
    return;

  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return;

  auto newType = cast<RankedTensorType>(
      getTensorCandidateType(value, solver, blockedTensorValues));
  if (newType != tensorType)
    value.setType(newType);
}

static std::optional<Type>
getTensorConsensusType(ValueRange values, DataFlowSolver &solver,
                       const llvm::DenseSet<Value> &blockedTensorValues) {
  if (values.empty())
    return std::nullopt;

  std::optional<Type> consensusType;
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      return std::nullopt;
    Type candidateType =
        getTensorCandidateType(value, solver, blockedTensorValues);
    if (!consensusType) {
      consensusType = candidateType;
      continue;
    }
    if (*consensusType != candidateType)
      return std::nullopt;
  }
  return consensusType;
}

static void appendRankedTensorValue(Value value,
                                    SmallVectorImpl<Value> &values) {
  if (value && isa<RankedTensorType>(value.getType()))
    values.push_back(value);
}

static scf::YieldOp getForYield(scf::ForOp forOp) {
  return dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
}

static void collectForCarrierPredecessors(scf::ForOp forOp, unsigned index,
                                          SmallVectorImpl<Value> &values) {
  if (index < forOp.getInitArgs().size())
    appendRankedTensorValue(forOp.getInitArgs()[index], values);
  if (auto yieldOp = getForYield(forOp))
    if (index < yieldOp.getNumOperands())
      appendRankedTensorValue(yieldOp.getOperand(index), values);
}

static void collectIfResultPredecessors(scf::IfOp ifOp, unsigned index,
                                        SmallVectorImpl<Value> &values) {
  if (auto thenYield = ifOp.thenYield())
    if (index < thenYield.getNumOperands())
      appendRankedTensorValue(thenYield.getOperand(index), values);
  if (auto elseYield = ifOp.elseYield())
    if (index < elseYield.getNumOperands())
      appendRankedTensorValue(elseYield.getOperand(index), values);
}

static llvm::DenseSet<Value>
computeBlockedTensorValues(tt::FuncOp func, DataFlowSolver &solver) {
  llvm::DenseSet<Value> blockedValues;
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](scf::ForOp forOp) {
      for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (!isa<RankedTensorType>(iterArg.getType()))
          continue;

        SmallVector<Value> predecessorValues;
        collectForCarrierPredecessors(forOp, index, predecessorValues);
        if (predecessorValues.empty())
          continue;
        if (getTensorConsensusType(ValueRange(predecessorValues), solver,
                                   blockedValues))
          continue;

        changed |= blockedValues.insert(iterArg).second;
        if (index < forOp.getNumResults())
          changed |= blockedValues.insert(forOp.getResult(index)).second;
        for (Value predecessorValue : predecessorValues)
          changed |= blockedValues.insert(predecessorValue).second;
      }
    });
    func.walk([&](scf::IfOp ifOp) {
      for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;

        SmallVector<Value> predecessorValues;
        collectIfResultPredecessors(ifOp, index, predecessorValues);
        if (predecessorValues.empty())
          continue;
        if (getTensorConsensusType(ValueRange(predecessorValues), solver,
                                   blockedValues))
          continue;

        changed |= blockedValues.insert(result).second;
        for (Value predecessorValue : predecessorValues)
          changed |= blockedValues.insert(predecessorValue).second;
      }
    });
  }
  return blockedValues;
}

static void updateTensorRegionBranchTypes(
    tt::FuncOp func, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedTensorValues) {
  func.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
    for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!isa<RankedTensorType>(iterArg.getType()))
        continue;

      SmallVector<Value> predecessorValues;
      collectForCarrierPredecessors(forOp, index, predecessorValues);
      std::optional<Type> consensusType = getTensorConsensusType(
          ValueRange(predecessorValues), solver, blockedTensorValues);
      if (!consensusType)
        continue;
      if (iterArg.getType() != *consensusType)
        iterArg.setType(*consensusType);
      if (index < forOp.getNumResults() &&
          forOp.getResult(index).getType() != *consensusType)
        forOp.getResult(index).setType(*consensusType);
    }
  });
  func.walk<WalkOrder::PostOrder>([&](scf::IfOp ifOp) {
    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isa<RankedTensorType>(result.getType()))
        continue;

      SmallVector<Value> predecessorValues;
      collectIfResultPredecessors(ifOp, index, predecessorValues);
      std::optional<Type> consensusType = getTensorConsensusType(
          ValueRange(predecessorValues), solver, blockedTensorValues);
      if (!consensusType || result.getType() == *consensusType)
        continue;
      result.setType(*consensusType);
    }
  });
}

static LogicalResult propagateExplicitLayoutContracts(tt::FuncOp func) {
  bool hasConstraint = false;
  func.walk([&](Operation *op) {
    if (isa<gluon_dialect::RequireLayoutOp, gluon_dialect::ReleaseLayoutOp>(op))
      hasConstraint = true;
  });
  if (!hasConstraint)
    return success();

  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<gluon_layout::LayoutBackwardPropagation>(symbolTable);
  solver.load<gluon_layout::LayoutForwardPropagation>();
  solver.load<gluon_layout::TensorBackwardPropagation>(symbolTable);
  if (failed(solver.initializeAndRun(func)))
    return failure();

  llvm::DenseSet<Value> blockedTensorValues =
      computeBlockedTensorValues(func, solver);

  WalkResult rewriteWalk = func.walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (failed(rewriteMemDescValueFromLattice(arg, solver, op)))
            return WalkResult::interrupt();
        }
      }
    }

    for (Value result : op->getResults()) {
      if (isa<ttg::MemDescType>(result.getType())) {
        if (failed(rewriteMemDescValueFromLattice(result, solver, op)))
          return WalkResult::interrupt();
      } else {
        rewriteTensorValueFromLattice(result, solver, blockedTensorValues);
      }
    }
    return WalkResult::advance();
  });
  if (rewriteWalk.wasInterrupted())
    return failure();

  updateTensorRegionBranchTypes(func, solver, blockedTensorValues);
  return success();
}

static LogicalResult cleanupSetAutoLayoutOps(tt::FuncOp func) {
  SmallVector<gluon_dialect::SetAutoLayoutOp> ops;
  func.walk([&](gluon_dialect::SetAutoLayoutOp op) { ops.push_back(op); });
  for (auto op : ops) {
    Value src = op.getSrc();
    if (src.getType() == op.getType()) {
      op.replaceAllUsesWith(src);
      op.erase();
      continue;
    }
    if (!isa<RankedTensorType>(src.getType()))
      return op.emitError() << "cannot materialize non-tensor set_auto_layout";
    OpBuilder builder(op);
    auto converted =
        builder.create<ttg::ConvertLayoutOp>(op.getLoc(), op.getType(), src);
    op.replaceAllUsesWith(converted.getResult());
    op.erase();
  }
  return success();
}

static LogicalResult cleanupRequireReleaseOps(tt::FuncOp func) {
  SmallVector<Operation *> ops;
  func.walk([&](Operation *op) {
    if (isa<gluon_dialect::RequireLayoutOp, gluon_dialect::ReleaseLayoutOp>(op))
      ops.push_back(op);
  });

  for (Operation *op : ops) {
    Value src = op->getOperand(0);
    Value result = op->getResult(0);
    if (src.getType() == result.getType()) {
      result.replaceAllUsesWith(src);
      op->erase();
      continue;
    }

    if (isa<ttg::MemDescType>(result.getType()))
      return op->emitError()
             << "memdesc layout contract was not absorbed by propagation";

    OpBuilder builder(op);
    auto converted = builder.create<ttg::ConvertLayoutOp>(
        op->getLoc(), result.getType(), src);
    result.replaceAllUsesWith(converted.getResult());
    op->erase();
  }
  return success();
}

static LogicalResult materializeExpandDimsLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](tt::ExpandDimsOp expandOp) {
    auto srcTy = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
    auto resultTy = dyn_cast<RankedTensorType>(expandOp.getType());
    if (!srcTy || !resultTy || !srcTy.getEncoding() ||
        !resultTy.getEncoding() || hasAutoEncoding(expandOp.getSrc()) ||
        hasAutoEncoding(expandOp.getResult()))
      return WalkResult::advance();

    auto parentEncoding =
        dyn_cast<ttg::DistributedEncodingTrait>(resultTy.getEncoding());
    if (!parentEncoding)
      return WalkResult::advance();

    Attribute requiredSrcEncoding = ttg::SliceEncodingAttr::get(
        expandOp.getContext(), expandOp.getAxis(), parentEncoding);
    if (unwrapNoVerifyEncoding(srcTy.getEncoding()) == requiredSrcEncoding)
      return WalkResult::advance();

    OpBuilder builder(expandOp);
    auto targetTy = RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), requiredSrcEncoding);
    auto converted = builder.create<ttg::ConvertLayoutOp>(
        expandOp.getLoc(), targetTy, expandOp.getSrc());
    expandOp->setOperand(0, converted.getResult());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult materializeAddPtrLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](tt::AddPtrOp addPtrOp) {
    auto ptrTy = dyn_cast<RankedTensorType>(addPtrOp.getPtr().getType());
    auto offsetTy =
        dyn_cast<RankedTensorType>(addPtrOp.getOffset().getType());
    auto resultTy = dyn_cast<RankedTensorType>(addPtrOp.getType());
    if (!ptrTy || !offsetTy || !resultTy || !resultTy.getEncoding() ||
        hasAutoEncoding(addPtrOp.getPtr()) ||
        hasAutoEncoding(addPtrOp.getOffset()) ||
        hasAutoEncoding(addPtrOp.getResult()))
      return WalkResult::advance();

    Attribute targetEncoding = unwrapNoVerifyEncoding(resultTy.getEncoding());
    OpBuilder builder(addPtrOp);
    auto convertOperand = [&](Value operand,
                              MutableOperandRange mutableOperand) {
      auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTy || !operandTy.getEncoding() ||
          unwrapNoVerifyEncoding(operandTy.getEncoding()) == targetEncoding)
        return;
      auto targetTy = RankedTensorType::get(
          operandTy.getShape(), operandTy.getElementType(), targetEncoding);
      auto converted = builder.create<ttg::ConvertLayoutOp>(
          addPtrOp.getLoc(), targetTy, operand);
      mutableOperand.assign(converted.getResult());
    };

    convertOperand(addPtrOp.getPtr(), addPtrOp.getPtrMutable());
    convertOperand(addPtrOp.getOffset(), addPtrOp.getOffsetMutable());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult materializeStoreLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](tt::StoreOp storeOp) {
    auto ptrTy = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());
    if (!ptrTy || !ptrTy.getEncoding() || hasAutoEncoding(storeOp.getPtr()))
      return WalkResult::advance();

    Attribute targetEncoding = unwrapNoVerifyEncoding(ptrTy.getEncoding());
    OpBuilder builder(storeOp);

    auto convertOperand = [&](Value operand,
                              MutableOperandRange mutableOperand) {
      auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTy || !operandTy.getEncoding() || hasAutoEncoding(operand) ||
          unwrapNoVerifyEncoding(operandTy.getEncoding()) == targetEncoding)
        return;
      auto targetTy = RankedTensorType::get(
          operandTy.getShape(), operandTy.getElementType(), targetEncoding);
      auto converted = builder.create<ttg::ConvertLayoutOp>(
          storeOp.getLoc(), targetTy, operand);
      mutableOperand.assign(converted.getResult());
    };

    convertOperand(storeOp.getValue(), storeOp.getValueMutable());
    if (storeOp.getMask())
      convertOperand(storeOp.getMask(), storeOp.getMaskMutable());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

class FoldRetaggedLocalAllocLoad
    : public OpRewritePattern<ttg::LocalLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalLoadOp localLoadOp,
                                PatternRewriter &rewriter) const override {
    auto allocOp = localLoadOp.getSrc().getDefiningOp<ttg::LocalAllocOp>();
    if (!allocOp || !allocOp.getSrc() || localLoadOp.getToken())
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
    if (!resultType ||
        !gluon_layout::isSupportedDotConstraintEncoding(
            resultType.getEncoding()))
      return failure();

    if (allocOp.getSrc().getType() == localLoadOp.getType()) {
      rewriter.replaceOp(localLoadOp, allocOp.getSrc());
      return success();
    }

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        localLoadOp, localLoadOp.getType(), allocOp.getSrc());
    return success();
  }
};

class FoldLocalAllocLoadFallback
    : public OpRewritePattern<ttg::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value src = allocOp.getSrc();
    if (!src || !isa<RankedTensorType>(src.getType()))
      return failure();

    SmallVector<ttg::LocalLoadOp> loads;
    for (Operation *user : allocOp->getUsers()) {
      auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user);
      if (!localLoadOp || localLoadOp.getToken())
        return failure();
      auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
      if (!resultType ||
          !gluon_layout::isSupportedDotConstraintEncoding(
              resultType.getEncoding()))
        return failure();
      loads.push_back(localLoadOp);
    }
    if (loads.empty())
      return failure();

    for (ttg::LocalLoadOp localLoadOp : loads) {
      rewriter.setInsertionPoint(localLoadOp);
      Value replacement = src;
      if (src.getType() != localLoadOp.getType())
        replacement = rewriter.create<ttg::ConvertLayoutOp>(
            localLoadOp.getLoc(), localLoadOp.getType(), src);
      rewriter.replaceOp(localLoadOp, replacement);
    }
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    return success();
  }
};

static LogicalResult foldLocalAllocLoadFallbacks(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  patterns.add<FoldRetaggedLocalAllocLoad>(module.getContext());
  patterns.add<FoldLocalAllocLoadFallback>(module.getContext());
  return applyPatternsGreedily(module, std::move(patterns));
}

static std::optional<SmallVector<int64_t>>
computeLogicalSubTensorIndexFromOffsets(RankedTensorType sourceTy,
                                        RankedTensorType subTensorTy,
                                        ArrayRef<int64_t> offsets,
                                        Location loc) {
  if (sourceTy.getRank() != subTensorTy.getRank() ||
      static_cast<int64_t>(offsets.size()) != sourceTy.getRank()) {
    emitError(loc) << "register slice rank mismatch";
    return std::nullopt;
  }

  SmallVector<int64_t> subTensorIdx;
  for (int64_t i = 0; i < sourceTy.getRank(); ++i) {
    int64_t sourceDim = sourceTy.getShape()[i];
    int64_t subDim = subTensorTy.getShape()[i];
    if (ShapedType::isDynamic(sourceDim) || ShapedType::isDynamic(subDim) ||
        subDim <= 0 || offsets[i] < 0 || offsets[i] + subDim > sourceDim ||
        offsets[i] % subDim != 0) {
      emitError(loc) << "unsupported Gluon register subview";
      return std::nullopt;
    }
    subTensorIdx.push_back(offsets[i] / subDim);
  }
  return subTensorIdx;
}

static LogicalResult rewriteGluonTensorGlueOps(tt::FuncOp func) {
  SmallVector<gluon_dialect::ExtractSliceOp> extractOps;
  SmallVector<gluon_dialect::InsertSliceOp> insertOps;
  func.walk([&](gluon_dialect::ExtractSliceOp op) { extractOps.push_back(op); });
  func.walk([&](gluon_dialect::InsertSliceOp op) { insertOps.push_back(op); });

  for (auto op : extractOps) {
    if (hasAutoEncoding(op.getSource()) || hasAutoEncoding(op.getResult()))
      continue;
    auto sourceTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!sourceTy || !resultTy)
      continue;
    auto subTensorIdx =
        computeLogicalSubTensorIndexFromOffsets(sourceTy, resultTy,
                                                op.getOffsets(), op.getLoc());
    if (!subTensorIdx)
      return failure();
    auto indices = calExtractTensorIdx(sourceTy, resultTy, *subTensorIdx);
    OpBuilder builder(op);
    auto replacement = builder.create<ttg::ExtractTensorOp>(
        op.getLoc(), op.getResult().getType(), op.getSource(),
        builder.getDenseI64ArrayAttr(indices.first),
        builder.getDenseI64ArrayAttr(indices.second));
    op.getResult().replaceAllUsesWith(replacement.getResult());
    op.erase();
  }

  for (auto op : insertOps) {
    if (hasAutoEncoding(op.getBase()) || hasAutoEncoding(op.getUpdate()) ||
        hasAutoEncoding(op.getResult()))
      continue;
    auto baseTy = dyn_cast<RankedTensorType>(op.getBase().getType());
    auto updateTy = dyn_cast<RankedTensorType>(op.getUpdate().getType());
    if (!baseTy || !updateTy)
      continue;
    auto subTensorIdx =
        computeLogicalSubTensorIndexFromOffsets(baseTy, updateTy,
                                                op.getOffsets(), op.getLoc());
    if (!subTensorIdx)
      return failure();
    auto indices = calExtractTensorIdx(baseTy, updateTy, *subTensorIdx);
    OpBuilder builder(op);
    auto replacement = builder.create<ttg::InsertTensorOp>(
        op.getLoc(), op.getResult().getType(), op.getBase(), op.getUpdate(),
        builder.getDenseI64ArrayAttr(indices.first),
        builder.getDenseI64ArrayAttr(indices.second));
    op.getResult().replaceAllUsesWith(replacement.getResult());
    op.erase();
  }
  return success();
}

static void resolveNoVerifyEncodings(tt::FuncOp func) {
  auto updateValue = [&](Value value) {
    if (auto encoding = getTypeEncoding(value.getType()))
      setValueTypeWithEncoding(value, encoding);
  };
  func.walk([&](Operation *op) {
    for (Value result : op->getResults())
      updateValue(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          updateValue(arg);
  });
}

static LogicalResult verifyNoAutoEncodings(ModuleOp module) {
  return gluon_dialect::doubleCheckEncodings(module, isAutoEncodingTensorType);
}

static bool hasNoVerifyEncoding(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<gluon_dialect::NoVerifyEncodingAttr>(tensorTy.getEncoding());
  if (auto memdescTy = dyn_cast<ttg::MemDescType>(type))
    return isa<gluon_dialect::NoVerifyEncodingAttr>(memdescTy.getEncoding());
  return false;
}

static LogicalResult verifyNoNoVerifyEncodings(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    for (Type type : op->getResultTypes()) {
      if (hasNoVerifyEncoding(type)) {
        op->emitError() << "unresolved gluon.no_verify_encoding result type";
        return WalkResult::interrupt();
      }
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Type type : block.getArgumentTypes()) {
          if (hasNoVerifyEncoding(type)) {
            op->emitError()
                << "unresolved gluon.no_verify_encoding block argument type";
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult verifyNoResidualLayoutContractOps(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (isa<gluon_dialect::SetAutoLayoutOp, gluon_dialect::RequireLayoutOp,
            gluon_dialect::ReleaseLayoutOp>(op)) {
      op->emitError() << "unresolved gluon layout contract op before RLC";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static bool isPowerOfTwo(unsigned value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static std::optional<SmallVector<unsigned>>
getStaticSubviewFactors(RankedTensorType fullTy, RankedTensorType subTy,
                        ArrayRef<int64_t> offsets) {
  if (fullTy.getRank() != subTy.getRank() ||
      offsets.size() != static_cast<size_t>(fullTy.getRank()))
    return std::nullopt;

  SmallVector<unsigned> factors;
  for (int64_t i = 0; i < fullTy.getRank(); ++i) {
    int64_t fullDim = fullTy.getShape()[i];
    int64_t subDim = subTy.getShape()[i];
    if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
        fullDim <= 0 || subDim <= 0 || fullDim % subDim != 0 ||
        offsets[i] < 0 || offsets[i] % subDim != 0 ||
        offsets[i] + subDim > fullDim)
      return std::nullopt;
    factors.push_back(static_cast<unsigned>(fullDim / subDim));
  }
  return factors;
}

static Attribute inferBlockedSubEncoding(RankedTensorType fullTy,
                                         RankedTensorType subTy,
                                         ArrayRef<int64_t> offsets,
                                         Attribute fullEncoding) {
  auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(fullEncoding);
  if (!blocked)
    return fullEncoding;
  std::optional<SmallVector<unsigned>> factors =
      getStaticSubviewFactors(fullTy, subTy, offsets);
  if (!factors)
    return fullEncoding;

  SmallVector<unsigned> sizePerThread(blocked.getSizePerThread());
  bool changed = false;
  for (auto [i, factor] : llvm::enumerate(*factors)) {
    if (factor <= 1)
      continue;
    unsigned adjusted = (sizePerThread[i] + factor - 1) / factor;
    adjusted = std::max<unsigned>(adjusted, 1);
    if (!isPowerOfTwo(adjusted))
      return fullEncoding;
    changed |= adjusted != sizePerThread[i];
    sizePerThread[i] = adjusted;
  }
  if (!changed)
    return fullEncoding;
  return ttg::BlockedEncodingAttr::get(
      fullTy.getContext(), sizePerThread, blocked.getThreadsPerWarp(),
      blocked.getWarpsPerCTA(), blocked.getOrder(), blocked.getCTALayout());
}

static Attribute inferBlockedFullEncoding(RankedTensorType fullTy,
                                          RankedTensorType subTy,
                                          ArrayRef<int64_t> offsets,
                                          Attribute subEncoding) {
  auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(subEncoding);
  if (!blocked)
    return subEncoding;
  std::optional<SmallVector<unsigned>> factors =
      getStaticSubviewFactors(fullTy, subTy, offsets);
  if (!factors)
    return subEncoding;

  SmallVector<unsigned> sizePerThread(blocked.getSizePerThread());
  bool changed = false;
  for (auto [i, factor] : llvm::enumerate(*factors)) {
    if (factor <= 1)
      continue;
    unsigned adjusted = sizePerThread[i] * factor;
    if (!isPowerOfTwo(adjusted))
      return subEncoding;
    changed |= adjusted != sizePerThread[i];
    sizePerThread[i] = adjusted;
  }
  if (!changed)
    return subEncoding;
  return ttg::BlockedEncodingAttr::get(
      fullTy.getContext(), sizePerThread, blocked.getThreadsPerWarp(),
      blocked.getWarpsPerCTA(), blocked.getOrder(), blocked.getCTALayout());
}

static std::optional<RankedTensorType>
inferTTGExtractResultType(Value source, RankedTensorType sourceTy,
                          RankedTensorType subShapeTy,
                          Attribute initialSubEncoding,
                          ArrayRef<int64_t> logicalSubTensorIdx,
                          Location loc) {
  auto indices =
      calExtractTensorIdx(sourceTy, subShapeTy, logicalSubTensorIdx);

  ttg::ExtractTensorOp::Properties properties;
  properties.ctaIdx =
      DenseI64ArrayAttr::get(sourceTy.getContext(), indices.first);
  properties.elemIdx =
      DenseI64ArrayAttr::get(sourceTy.getContext(), indices.second);

  SmallVector<Type, 1> inferredTypes;
  Type oldType = source.getType();
  source.setType(sourceTy);
  auto result = ttg::ExtractTensorOp::inferReturnTypes(
      sourceTy.getContext(), loc, ValueRange(source), DictionaryAttr(),
      OpaqueProperties(&properties), RegionRange(), inferredTypes);
  source.setType(oldType);

  if (failed(result) || inferredTypes.empty())
    return std::nullopt;

  auto inferredTy = dyn_cast<RankedTensorType>(inferredTypes[0]);
  if (!inferredTy)
    return std::nullopt;

  if (initialSubEncoding && inferredTy.getEncoding() == initialSubEncoding)
    return inferredTy;
  return inferredTy;
}

static std::optional<RankedTensorType>
inferStableTTGExtractResultType(Value source, RankedTensorType sourceTy,
                                RankedTensorType subShapeTy,
                                Attribute initialSubEncoding,
                                ArrayRef<int64_t> logicalSubTensorIdx,
                                Location loc) {
  Attribute candidateEncoding = initialSubEncoding;
  std::optional<RankedTensorType> inferredTy;

  for (int i = 0; i < 4; ++i) {
    auto candidateSubTy = subShapeTy.cloneWithEncoding(candidateEncoding);
    inferredTy = inferTTGExtractResultType(source, sourceTy, candidateSubTy,
                                           candidateEncoding,
                                           logicalSubTensorIdx, loc);
    if (!inferredTy)
      return std::nullopt;
    Attribute inferredEncoding = inferredTy->getEncoding();
    if (inferredEncoding == candidateEncoding)
      return inferredTy;
    candidateEncoding = inferredEncoding;
  }
  return inferredTy;
}

static SmallVector<SmallVector<unsigned, 2>>
getC500GlobalMemoryOrderCandidates(Attribute resultEncoding) {
  SmallVector<SmallVector<unsigned, 2>> orders;
  auto addOrder = [&](ArrayRef<unsigned> order) {
    if (order.size() != 2)
      return;
    SmallVector<unsigned, 2> candidate(order.begin(), order.end());
    for (ArrayRef<unsigned> existing : orders)
      if (existing == ArrayRef<unsigned>(candidate))
        return;
    orders.push_back(candidate);
  };

  if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(resultEncoding))
    addOrder(blocked.getOrder());
  addOrder(ArrayRef<unsigned>{1, 0});
  addOrder(ArrayRef<unsigned>{0, 1});
  return orders;
}

static std::optional<ttg::BlockedEncodingAttr>
tryExtractParentBlockedEncoding(gluon_dialect::ExtractSliceOp op,
                                Attribute resultEncoding,
                                ttg::BlockedEncodingAttr parentEncoding) {
  auto sourceShapeTy = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto resultShapeTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!sourceShapeTy || !resultShapeTy)
    return std::nullopt;

  auto sourceTy = sourceShapeTy.cloneWithEncoding(parentEncoding);
  auto subTensorIdx =
      computeLogicalSubTensorIndexFromOffsets(sourceTy, resultShapeTy,
                                              op.getOffsets(), op.getLoc());
  if (!subTensorIdx)
    return std::nullopt;
  auto inferredTy = inferStableTTGExtractResultType(
      op.getSource(), sourceTy, resultShapeTy, resultEncoding, *subTensorIdx,
      op.getLoc());
  if (inferredTy && inferredTy->getEncoding() == resultEncoding &&
      inferredTy->getShape().equals(resultShapeTy.getShape()))
    return parentEncoding;
  return std::nullopt;
}

static std::optional<ttg::BlockedEncodingAttr>
inferExtractParentBlockedEncoding(gluon_dialect::ExtractSliceOp op,
                                  Attribute resultEncoding) {
  if (auto resultBlocked = dyn_cast<ttg::BlockedEncodingAttr>(resultEncoding)) {
    auto parentEncoding = c500::getC500GlobalBlockedEncodingLike(
        op.getSource(), op, resultBlocked);
    if (parentEncoding)
      if (auto stable = tryExtractParentBlockedEncoding(
              op, resultEncoding, *parentEncoding))
        return stable;
  }

  for (auto order : getC500GlobalMemoryOrderCandidates(resultEncoding)) {
    auto parentEncoding =
        c500::getC500GlobalBlockedEncoding(op.getSource(), op, order);
    if (!parentEncoding)
      continue;
    if (auto stable =
            tryExtractParentBlockedEncoding(op, resultEncoding, *parentEncoding))
      return stable;
  }
  return std::nullopt;
}

static Attribute
inferExtractResultEncodingWithC500(gluon_dialect::ExtractSliceOp op,
                                   Attribute sourceEncoding) {
  if (!sourceEncoding)
    return {};

  auto sourceShapeTy = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto resultShapeTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!sourceShapeTy || !resultShapeTy)
    return {};

  auto sourceTy = sourceShapeTy.cloneWithEncoding(sourceEncoding);
  auto subTensorIdx =
      computeLogicalSubTensorIndexFromOffsets(sourceTy, resultShapeTy,
                                              op.getOffsets(), op.getLoc());
  if (!subTensorIdx)
    return {};
  auto inferredTy = inferStableTTGExtractResultType(
      op.getSource(), sourceTy, resultShapeTy, sourceEncoding, *subTensorIdx,
      op.getLoc());
  if (!inferredTy || !inferredTy->getShape().equals(resultShapeTy.getShape()))
    return {};
  return inferredTy->getEncoding();
}

static Attribute
inferExtractSourceEncodingWithC500(gluon_dialect::ExtractSliceOp op,
                                   Attribute resultEncoding) {
  if (!resultEncoding)
    return {};
  resultEncoding = unwrapNoVerifyEncoding(resultEncoding);

  if (isa<ttg::BlockedEncodingAttr>(resultEncoding)) {
    auto sourceTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    if (sourceTy) {
      Attribute sourceEncoding = unwrapNoVerifyEncoding(sourceTy.getEncoding());
      if (sourceEncoding &&
          !isa<gluon_dialect::AutoEncodingAttr>(sourceEncoding)) {
        Attribute inferredResult =
            inferExtractResultEncodingWithC500(op, sourceEncoding);
        if (unwrapNoVerifyEncoding(inferredResult) == resultEncoding)
          return sourceEncoding;
      }
    }

    auto parentEncoding = inferExtractParentBlockedEncoding(op, resultEncoding);
    return parentEncoding ? Attribute(*parentEncoding) : Attribute();
  }

  if (isa<ttg::MACAMmaEncodingAttr, ttg::DotOperandEncodingAttr>(
          resultEncoding)) {
    auto sourceTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!sourceTy || !resultTy)
      return {};

    Attribute sourceEncoding = unwrapNoVerifyEncoding(sourceTy.getEncoding());
    if (sourceEncoding &&
        !isa<gluon_dialect::AutoEncodingAttr>(sourceEncoding)) {
      Attribute inferredResult =
          inferExtractResultEncodingWithC500(op, sourceEncoding);
      if (unwrapNoVerifyEncoding(inferredResult) == resultEncoding)
        return sourceEncoding;
    }

    if (sourceTy.getShape().equals(resultTy.getShape()))
      return resultEncoding;

    if (isa<ttg::DotOperandEncodingAttr>(resultEncoding) &&
        sourceTy.getRank() == resultTy.getRank() &&
        sourceTy.getElementType() == resultTy.getElementType()) {
      bool sourceCoversResult = true;
      for (auto [sourceDim, resultDim] :
           llvm::zip(sourceTy.getShape(), resultTy.getShape())) {
        if (ShapedType::isDynamic(sourceDim) ||
            ShapedType::isDynamic(resultDim) || sourceDim < resultDim) {
          sourceCoversResult = false;
          break;
        }
      }
      if (sourceCoversResult)
        return resultEncoding;
    }
  }

  return {};
}

static Attribute
inferInsertSubEncodingWithC500(gluon_dialect::InsertSliceOp op,
                               Attribute fullEncoding) {
  if (!fullEncoding)
    return {};

  auto baseShapeTy = dyn_cast<RankedTensorType>(op.getBase().getType());
  auto updateShapeTy = dyn_cast<RankedTensorType>(op.getUpdate().getType());
  if (!baseShapeTy || !updateShapeTy)
    return {};

  auto baseTy = baseShapeTy.cloneWithEncoding(fullEncoding);
  auto subTensorIdx =
      computeLogicalSubTensorIndexFromOffsets(baseTy, updateShapeTy,
                                              op.getOffsets(), op.getLoc());
  if (!subTensorIdx)
    return {};
  auto inferredTy = inferStableTTGExtractResultType(
      op.getBase(), baseTy, updateShapeTy, fullEncoding, *subTensorIdx,
      op.getLoc());
  if (!inferredTy || !inferredTy->getShape().equals(updateShapeTy.getShape()))
    return {};
  return inferredTy->getEncoding();
}

static gluon_dialect::LayoutInferenceHooks getC500LayoutInferenceHooks() {
  gluon_dialect::LayoutInferenceHooks hooks;
  hooks.inferExtractResult =
      [](gluon_dialect::ExtractSliceOp op, Attribute sourceEncoding) {
        return inferExtractResultEncodingWithC500(op, sourceEncoding);
      };
  hooks.inferExtractSource =
      [](gluon_dialect::ExtractSliceOp op, Attribute resultEncoding) {
        return inferExtractSourceEncodingWithC500(op, resultEncoding);
      };
  hooks.inferInsertSub =
      [](gluon_dialect::InsertSliceOp op, Attribute fullEncoding) {
        return inferInsertSubEncodingWithC500(op, fullEncoding);
      };
  return hooks;
}

class TritonMETAXGPUGluonPropagateLayoutPass
    : public impl::TritonMETAXGPUGluonPropagateLayoutBase<
          TritonMETAXGPUGluonPropagateLayoutPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto hooks = getC500LayoutInferenceHooks();
    for (auto func : module.getOps<tt::FuncOp>()) {
      SmallVector<std::pair<Value, Attribute>> seeds;
      collectLayoutSeeds(func, seeds);
      if (failed(gluon_dialect::inferLayout(func, isAutoEncodingTensorType,
                                            seeds, hooks))) {
        signalPassFailure();
        return;
      }
      resolveNoVerifyEncodings(func);
      if (failed(propagateExplicitLayoutContracts(func)) ||
          failed(materializeExpandDimsLayouts(func)) ||
          failed(materializeAddPtrLayouts(func)) ||
          failed(cleanupSetAutoLayoutOps(func)) ||
          failed(cleanupRequireReleaseOps(func)) ||
          failed(c500::materializeAsyncCopyFollowerLayouts(func)) ||
          failed(c500::verifyAsyncCopyLayouts(func)) ||
          failed(materializeStoreLayouts(func)) ||
          failed(rewriteGluonTensorGlueOps(func))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(foldLocalAllocLoadFallbacks(module)) ||
        failed(verifyNoResidualLayoutContractOps(module)) ||
        failed(verifyNoAutoEncodings(module)) ||
        failed(verifyNoNoVerifyEncodings(module)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTritonMETAXGPUGluonPropagateLayoutPass() {
  return std::make_unique<TritonMETAXGPUGluonPropagateLayoutPass>();
}

} // namespace mlir
