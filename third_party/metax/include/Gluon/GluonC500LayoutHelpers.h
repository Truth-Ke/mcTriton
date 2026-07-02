#ifndef TRITON_METAX_GLUON_C500_LAYOUT_HELPERS_H
#define TRITON_METAX_GLUON_C500_LAYOUT_HELPERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir::triton::gpu::metax::gluon::c500 {

Value getMemDescViewSource(Operation *op);
Value getMemDescRoot(Value value);
void collectMemDescAliasValues(Value root, llvm::SetVector<Value> &aliases);

template <typename... ProducerOps>
bool isFedByAnyMemDescUser(Value memdesc) {
  Value root = getMemDescRoot(memdesc);
  if (!root)
    return false;
  llvm::SetVector<Value> aliases;
  collectMemDescAliasValues(root, aliases);
  for (Value alias : aliases)
    for (Operation *user : alias.getUsers())
      if (isa<ProducerOps...>(user))
        return true;
  return false;
}

unsigned getTensorElementOrPointeeBitWidth(RankedTensorType tensorTy);

SmallVector<unsigned, 2>
chooseRank2ContiguousOrder(Value value,
                           triton::ModuleAxisInfoAnalysis *axisInfoAnalysis =
                               nullptr);

std::optional<triton::gpu::BlockedEncodingAttr>
getC500GlobalBlockedEncoding(Value value, Operation *anchor,
                             ArrayRef<unsigned> order);

std::optional<triton::gpu::BlockedEncodingAttr>
getC500GlobalBlockedEncodingLike(Value value, Operation *anchor,
                                 triton::gpu::BlockedEncodingAttr model);

std::optional<unsigned> getContiguityHint(Value value, unsigned dim);

/// Validates that subview materialization selects contiguous CTA/element
/// ranges. This is intentionally conservative for C500 async_copy because a
/// non-contiguous ttg.extract_tensor index sequence would describe a scattered
/// global/shared transfer even if the tensor encoding itself looks legal.
bool hasContiguousSubviewIndices(Value value);

bool isC500AsyncCopyContiguousSharedWrite(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

LogicalResult verifyC500AsyncCopyContiguousSharedWrite(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

} // namespace mlir::triton::gpu::metax::gluon::c500

#endif // TRITON_METAX_GLUON_C500_LAYOUT_HELPERS_H
