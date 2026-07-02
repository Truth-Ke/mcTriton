#ifndef TRITON_METAX_GLUON_C500_ASYNC_COPY_LAYOUT_H
#define TRITON_METAX_GLUON_C500_ASYNC_COPY_LAYOUT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>

namespace mlir::triton::gpu::metax::gluon::c500 {

struct DotConsumerInfo {
  triton::gpu::DotOperandEncodingAttr dotEnc;
  SmallVector<int64_t> tensorShape;
  Type elementType;
  std::optional<SmallVector<unsigned>> userSharedOrder;

  bool operator==(const DotConsumerInfo &rhs) const {
    return dotEnc == rhs.dotEnc && tensorShape == rhs.tensorShape &&
           elementType == rhs.elementType &&
           userSharedOrder == rhs.userSharedOrder;
  }
};

bool isFedByC500AsyncCopyProducer(Value memdesc);

FailureOr<Attribute> chooseAsyncCopySharedEncoding(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp, Value dstMemDesc,
    std::optional<DotConsumerInfo> dotConsumer);

LogicalResult materializeAsyncCopyFollowerLayouts(triton::FuncOp func);

LogicalResult verifyAsyncCopyLayouts(triton::FuncOp func);

} // namespace mlir::triton::gpu::metax::gluon::c500

#endif // TRITON_METAX_GLUON_C500_ASYNC_COPY_LAYOUT_H
