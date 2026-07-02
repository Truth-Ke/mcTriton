#include "Gluon/Analysis/GluonLayoutPropagation.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "metax-gluon-layout-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {

static FailureOr<Attribute> inferTransEncoding(Attribute encoding,
                                               ArrayRef<int64_t> shape,
                                               ArrayRef<int32_t> order,
                                               Location loc) {
  Dialect &dialect = encoding.getDialect();
  auto inferLayoutInterface =
      cast<::mlir::triton::DialectInferLayoutInterface>(&dialect);
  Attribute resultEncoding;
  if (failed(inferLayoutInterface->inferTransOpEncoding(encoding, shape, order,
                                                        resultEncoding, loc)))
    return failure();
  return resultEncoding;
}

static SmallVector<int32_t> invertPermutation(ArrayRef<int32_t> order) {
  SmallVector<int32_t> inverse(order.size());
  for (auto [i, dim] : llvm::enumerate(order))
    inverse[dim] = i;
  return inverse;
}

void LayoutEncoding::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  if (isUnknown()) {
    os << "<UNKNOWN>";
    return;
  }
  return getLayoutEncoding().print(os);
}

LayoutEncoding LayoutEncoding::join(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return LayoutEncoding::getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;
  return LayoutEncoding::getUnknownLayout();
}

LayoutEncoding LayoutEncoding::meet(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  return join(lhs, rhs);
}

LogicalResult LayoutBackwardPropagation::visitRegionInReverse(Operation *op) {
  for (Region &region : llvm::reverse(op->getRegions())) {
    for (Block &block : llvm::reverse(region)) {
      for (Operation &nestedOp : llvm::reverse(block)) {
        SmallVector<LayoutEncodingLattice *> operands;
        for (Value operand : nestedOp.getOperands())
          operands.push_back(getLatticeElement(operand));
        SmallVector<const LayoutEncodingLattice *> results;
        for (Value result : nestedOp.getResults())
          results.push_back(getLatticeElement(result));
        if (failed(visitOperation(&nestedOp, operands, results)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult LayoutBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
    ArrayRef<const LayoutEncodingLattice *> results) {
  if (isa<gluon_dialect::ReleaseLayoutOp>(op))
    return success();

  if (isa<RegionBranchOpInterface>(op))
    return visitRegionInReverse(op);

  if (auto transOp = dyn_cast<ttg::MemDescTransOp>(op)) {
    LayoutEncoding resultLayout = results[0]->getValue();
    if (resultLayout.isUninitialized() || resultLayout.isUnknown())
      return success();

    auto resultType = cast<ttg::MemDescType>(transOp.getType());
    auto inverseOrder = invertPermutation(transOp.getOrder());
    FailureOr<Attribute> inferred =
        inferTransEncoding(resultLayout.getLayoutEncoding(),
                           resultType.getShape(), inverseOrder, op->getLoc());
    if (failed(inferred))
      return failure();
    ChangeResult changed = operands[0]->meet(LayoutEncoding(*inferred));
    propagateIfChanged(operands[0], changed);
    return success();
  }

  if (auto reshapeOp = dyn_cast<ttg::MemDescReshapeOp>(op)) {
    LayoutEncoding resultLayout = results[0]->getValue();
    if (resultLayout.isUninitialized() || resultLayout.isUnknown())
      return success();

    auto srcType = cast<ttg::MemDescType>(reshapeOp.getSrc().getType());
    auto resultType = cast<ttg::MemDescType>(reshapeOp.getType());
    auto resultTypeWithLayout = ttg::MemDescType::get(
        resultType.getShape(), resultType.getElementType(),
        resultLayout.getLayoutEncoding(), resultType.getMemorySpace(),
        resultType.getMutableMemory(), resultType.getAllocShape());
    ttg::MemDescType inferredSrcType;
    if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
            op->getContext(), op->getLoc(), resultTypeWithLayout,
            srcType.getShape(), inferredSrcType)))
      return failure();
    ChangeResult changed =
        operands[0]->meet(LayoutEncoding(inferredSrcType.getEncoding()));
    propagateIfChanged(operands[0], changed);
    return success();
  }

  if (auto requireOp = dyn_cast<gluon_dialect::RequireLayoutOp>(op)) {
    if (isa<RankedTensorType>(requireOp.getType()))
      return success();
    Attribute layout = cast<ttg::MemDescType>(requireOp.getType()).getEncoding();
    ChangeResult changed = operands[0]->meet(LayoutEncoding(layout));
    propagateIfChanged(operands[0], changed);
    return success();
  }

  for (const auto resultLattice : results) {
    for (auto [i, operandLattice] : llvm::enumerate(operands)) {
      if (!isa<ttg::MemDescType>(op->getOpOperand(i).get().getType()))
        continue;
      ChangeResult changed = operandLattice->meet(resultLattice->getValue());
      propagateIfChanged(operandLattice, changed);
    }
  }
  return success();
}

void LayoutBackwardPropagation::visitBranchOperand(OpOperand &operand) {}

void LayoutBackwardPropagation::visitCallOperand(OpOperand &operand) {
  llvm_unreachable("Gluon layout propagation expects calls to be inlined");
}

void LayoutBackwardPropagation::setToExitState(LayoutEncodingLattice *) {}

void TensorLayout::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  if (isUnknown()) {
    os << "<UNKNOWN>";
    return;
  }
  return getLayoutEncoding().print(os);
}

TensorLayout TensorLayout::join(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  return meet(lhs, rhs);
}

TensorLayout TensorLayout::meet(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return TensorLayout::getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;
  return TensorLayout::getUnknownLayout();
}

static bool isTrackedTensorValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

static bool isAllowedTensorLayoutUser(Operation *op, unsigned operandIndex) {
  if (auto requireOp = dyn_cast<gluon_dialect::RequireLayoutOp>(op)) {
    if (!isa<RankedTensorType>(requireOp.getType()) || operandIndex != 0)
      return false;
    return isSupportedTensorConstraintEncoding(
        cast<RankedTensorType>(requireOp.getType()).getEncoding());
  }

  return isa<ttg::ConvertLayoutOp>(op) || isTransparentLayoutCarrierOp(op);
}

static bool isDotLocalAllocFallback(Operation *op, unsigned operandIndex) {
  auto allocOp = dyn_cast<ttg::LocalAllocOp>(op);
  if (!allocOp || operandIndex != 0 || !allocOp.getSrc())
    return false;
  if (allocOp->use_empty())
    return false;

  for (Operation *user : allocOp->getUsers()) {
    auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user);
    if (!localLoadOp)
      return false;
    auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
    if (!resultType ||
        !isSupportedDotConstraintEncoding(resultType.getEncoding()))
      return false;
  }
  return true;
}

static bool canRewriteTensorResult(Operation *op) {
  return isa<ttg::LocalLoadOp, RegionBranchOpInterface>(op);
}

LogicalResult TensorBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TensorLayoutLattice *> operands,
    ArrayRef<const TensorLayoutLattice *> results) {
  if (auto requireOp = dyn_cast<gluon_dialect::RequireLayoutOp>(op)) {
    if (!isa<RankedTensorType>(requireOp.getType()))
      return success();
    Attribute layout = cast<RankedTensorType>(requireOp.getType()).getEncoding();
    if (!isSupportedTensorConstraintEncoding(layout))
      return success();
    ChangeResult changed = operands[0]->meet(TensorLayout(layout));
    propagateIfChanged(operands[0], changed);
    return success();
  }

  if (isa<gluon_dialect::ReleaseLayoutOp>(op))
    return success();

  if (auto convertOp = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (!results.empty() && isTrackedTensorValue(convertOp.getSrc())) {
      const TensorLayout &resultState = results[0]->getValue();
      if (!resultState.isUnknown()) {
        ChangeResult changed = operands[0]->meet(resultState);
        propagateIfChanged(operands[0], changed);
      }
    }
  }

  if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
    if (!allocOp.getSrc() || !isTrackedTensorValue(allocOp.getSrc()) ||
        !isDotLocalAllocFallback(op, /*operandIndex=*/0))
      return success();

    TensorLayout state;
    for (Operation *user : allocOp->getUsers()) {
      auto localLoadOp = cast<ttg::LocalLoadOp>(user);
      auto resultType = cast<RankedTensorType>(localLoadOp.getType());
      state = TensorLayout::meet(state, TensorLayout(resultType.getEncoding()));
      state = TensorLayout::meet(
          state, getLatticeElement(localLoadOp.getResult())->getValue());
    }
    if (!state.isUninitialized()) {
      ChangeResult changed = operands[0]->meet(state);
      propagateIfChanged(operands[0], changed);
    }
    return success();
  }

  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (!isTrackedTensorValue(operand))
      continue;
    if (isAllowedTensorLayoutUser(op, index))
      continue;

    TensorLayout operandState = operands[index]->getValue();
    if (operandState.isUninitialized())
      continue;
    ChangeResult changed =
        operands[index]->meet(TensorLayout::getUnknownLayout());
    propagateIfChanged(operands[index], changed);
  }

  if (!canRewriteTensorResult(op)) {
    for (Value result : op->getResults()) {
      if (!isTrackedTensorValue(result))
        continue;
      auto *resultLattice = getLatticeElement(result);
      TensorLayout resultState = resultLattice->getValue();
      if (resultState.isUninitialized())
        continue;
      ChangeResult changed =
          resultLattice->meet(TensorLayout::getUnknownLayout());
      propagateIfChanged(resultLattice, changed);
    }
  }

  return success();
}

void TensorBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  if (!isTrackedTensorValue(operand.get()))
    return;
  if (isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(
          operand.getOwner()))
    return;

  auto *lattice = getLatticeElement(operand.get());
  TensorLayout state = lattice->getValue();
  if (state.isUninitialized())
    return;
  ChangeResult changed = lattice->meet(TensorLayout::getUnknownLayout());
  propagateIfChanged(lattice, changed);
}

void TensorBackwardPropagation::visitCallOperand(OpOperand &operand) {
  if (!isTrackedTensorValue(operand.get()))
    return;
  auto *lattice = getLatticeElement(operand.get());
  TensorLayout state = lattice->getValue();
  if (state.isUninitialized())
    return;
  ChangeResult changed = lattice->meet(TensorLayout::getUnknownLayout());
  propagateIfChanged(lattice, changed);
}

void TensorBackwardPropagation::setToExitState(TensorLayoutLattice *) {}

LogicalResult LayoutForwardPropagation::visitOperation(
    Operation *op, ArrayRef<const LayoutEncodingLattice *> operands,
    ArrayRef<LayoutEncodingLattice *> results) {
  if (isa<RegionBranchOpInterface>(op))
    return visitRegion(op);

  if (!isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
           ttg::MemDescSubsliceOp, ttg::MemDescTransOp, ttg::MemDescReshapeOp,
           ttg::LocalAllocOp>(op))
    return success();

  for (const auto [operandIdx, operandLattice] : llvm::enumerate(operands)) {
    if (!isa<ttg::MemDescType>(op->getOperand(operandIdx).getType()))
      continue;
    LayoutEncoding layout = operandLattice->getValue();

    if (auto transOp = dyn_cast<ttg::MemDescTransOp>(op)) {
      if (!layout.isUninitialized() && !layout.isUnknown()) {
        auto srcTy = cast<ttg::MemDescType>(transOp.getSrc().getType());
        FailureOr<Attribute> inferred = inferTransEncoding(
            layout.getLayoutEncoding(), srcTy.getShape(), transOp.getOrder(),
            op->getLoc());
        if (failed(inferred))
          return failure();
        layout = LayoutEncoding(*inferred);
      }
    }

    if (auto reshapeOp = dyn_cast<ttg::MemDescReshapeOp>(op)) {
      if (!layout.isUninitialized() && !layout.isUnknown()) {
        auto srcTy = cast<ttg::MemDescType>(reshapeOp.getSrc().getType());
        auto srcTyWithLayout = ttg::MemDescType::get(
            srcTy.getShape(), srcTy.getElementType(), layout.getLayoutEncoding(),
            srcTy.getMemorySpace(), srcTy.getMutableMemory(),
            srcTy.getAllocShape());
        ttg::MemDescType inferredResultType;
        auto dstTy = cast<ttg::MemDescType>(reshapeOp.getType());
        if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
                op->getContext(), op->getLoc(), srcTyWithLayout,
                dstTy.getShape(), inferredResultType)))
          return failure();
        layout = LayoutEncoding(inferredResultType.getEncoding());
      }
    }

    for (auto resultLattice : results) {
      ChangeResult changed = resultLattice->meet(layout);
      propagateIfChanged(resultLattice, changed);
    }
  }
  return success();
}

LogicalResult LayoutForwardPropagation::visitRegion(Operation *op) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        SmallVector<const LayoutEncodingLattice *> operands;
        for (Value operand : nestedOp.getOperands())
          operands.push_back(getLatticeElement(operand));
        SmallVector<LayoutEncodingLattice *> results;
        for (Value result : nestedOp.getResults())
          results.push_back(getLatticeElement(result));
        if (failed(visitOperation(&nestedOp, operands, results)))
          return failure();
      }
    }
  }
  return success();
}

void LayoutForwardPropagation::setToEntryState(LayoutEncodingLattice *) {}

} // namespace mlir::triton::gpu::metax::gluon
