add_triton_library(TritonMETAXGPUGluon
  Analysis/GluonLayoutPropagation.cpp
  GluonC500LayoutHelpers.cpp
  GluonC500AsyncCopyLayout.cpp
  GluonInsertRequireLayout.cpp
  GluonPropagateLayout.cpp

  DEPENDS
  TritonMETAXGPUGluonIncGen
  TritonGPUIR

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  GluonIR
  GluonTransforms
  TritonMETAXGPUTransforms
)

target_include_directories(TritonMETAXGPUGluon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMETAXGPUGluon PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
