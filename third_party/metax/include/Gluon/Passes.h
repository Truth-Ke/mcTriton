#ifndef TRITON_METAX_GLUON_PASSES_H
#define TRITON_METAX_GLUON_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DECL
#include "Gluon/Passes.h.inc"

std::unique_ptr<Pass> createTritonMETAXGPUGluonInsertRequireLayoutPass(
    int computeCapability = 80, bool storeCoalesce = false);

std::unique_ptr<Pass> createTritonMETAXGPUGluonPropagateLayoutPass();

#define GEN_PASS_REGISTRATION
#include "Gluon/Passes.h.inc"

} // namespace mlir

#endif // TRITON_METAX_GLUON_PASSES_H
