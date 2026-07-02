#ifndef TRITON_METAX_GLUON_PASSES
#define TRITON_METAX_GLUON_PASSES

include "mlir/Pass/PassBase.td"

def TritonMETAXGPUGluonInsertRequireLayout
    : Pass<"tritonmetaxgpu-gluon-insert-require-layout", "mlir::ModuleOp"> {
  let summary = "Insert explicit MetaX/C500 Gluon layout requirements";

  let description = [{
    Discovers implicit layout requirements from dot/local_load and C500
    async-copy dataflow, then records them as explicit gluon.require_layout
    contracts. This pass only synthesizes constraints; propagation, retagging,
    cleanup, and fallback conversion are owned by
    tritonmetaxgpu-gluon-propagate-layout.
  }];

  let constructor = "mlir::createTritonMETAXGPUGluonInsertRequireLayoutPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "MetaX compute capability">,
    Option<"storeCoalesce", "store-coalesce",
           "bool", /*default*/"false",
           "Use store-coalesced MetaX MMA layout policy">
  ];
}

def TritonMETAXGPUGluonPropagateLayout
    : Pass<"tritonmetaxgpu-gluon-propagate-layout", "mlir::ModuleOp"> {
  let summary = "Propagate explicit MetaX/C500 Gluon layout requirements";

  let description = [{
    Propagates explicit gluon.require_layout and gluon.release_layout
    contracts through tensor and memdesc dataflow, resolves Gluon
    AutoEncodingAttr to concrete TTG layouts, applies conservative region
    consensus, lowers residual tensor constraints to ttg.convert_layout, and
    lowers Gluon slice glue to TTG tensor slice ops.
  }];

  let constructor = "mlir::createTritonMETAXGPUGluonPropagateLayoutPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

#endif
