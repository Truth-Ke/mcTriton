# MetaX/C500 Gluon TLX-style Layout 大重构实施清单

日期：2026-07-01

本文档推翻旧的“小修小补”方案。新的方向是：mcTriton/Gluon 的 layout 自动推导尽量对齐 TLX 的 pass 名字、顺序、职责和代码骨架；旧 `GluonLayoutUtils.cpp` 中 seed/infer/materialize 混在一起的逻辑不作为演进基础。
编译cmd: {bash /datapool/kezengxiang/2026/mcTriton/maca/maca_tools/build_triton.sh     --llvm /datapool/kezengxiang/2026/third_party/metax_llvm_3.8.0/     -m ${MACA_PATH}}
编译和算子都要非沙箱跑！

本轮反思结论：

- 大方向正确：采用 TLX 两 pass 加 Analysis 的骨架，而不是继续扩展旧 Gluon layout 工具函数。
- 必须修正：旧 pass 不能作为 fallback 路径；本轮是直接替换。
- 必须补足：async_copy 不只是 src/mask/other 的 tensor layout 约束，还要像 TLX TDM 一样 anchor destination memdesc 的硬件兼容 shared encoding。
- 必须补足：C500 shared 写入连续性是 correctness invariant，不是优化项。
- 必须补足：layout 自动推导和 layout 优化要分层，避免在传播 pass 里塞过多 cost model。

核心原则：

- 尽量直接复制 TLX 的 `InsertRequireLayout.cpp`、`PropagateLayout.cpp`、`Analysis/LayoutPropagation.*`，再做 C500/Gluon 本地适配。
- 不实现 C500 没有的 TDM、TMEM、WarpSpecialize；这些只保留扩展接口和后续目标。
- C500 的 `ttg.async_copy_global_to_local` 替代 TLX/AMD TDM anchor 的位置，是当前最重要的硬件边界。
- C500 写 shared memory 必须连续写入，不能为了满足 dot layout 推出离散 shared 写入。
- `async_copy` 的 `src` / `mask` / `other` 必须使用同一套 tensor encoding。
- `slice` / `insert_slice` 是 subview 语义，parent/sub layout 要通过可解释的变换互推，不能简单粗暴全按 same-layout 处理。
- 不重新发明大而全 solver；直接沿用 TLX 的 SparseDataFlowAnalysis + lattice 思路。
- layout legality 和 layout optimization 分层：先求一个硬件正确的 layout，再让 RLC/MetaX 优化 pass 做局部择优。

## 0. TLX 母版和裁剪边界

必须优先参考并尽量复制：

- `/datapool/kezengxiang/2026/tlx/third_party/tlx/dialect/lib/Transforms/InsertRequireLayout.cpp`
- `/datapool/kezengxiang/2026/tlx/third_party/tlx/dialect/lib/Transforms/PropagateLayout.cpp`
- `/datapool/kezengxiang/2026/tlx/third_party/tlx/dialect/include/Analysis/LayoutPropagation.h`
- `/datapool/kezengxiang/2026/tlx/third_party/tlx/dialect/lib/Analysis/LayoutPropagation.cpp`
- `/datapool/kezengxiang/2026/tlx/third_party/tlx/dialect/include/Transforms/Passes.td`

当前不要复制或实现：

- `StorageAliasAllocation.cpp`
- `StorageAliasLowering.cpp`
- `StorageAliasSizeDefinition.cpp`
- `RewriteLocalAlias.cpp`
- `PrintTTGIRToTLX.cpp`
- `ResolvePlaceholderLayouts.cpp` 中 TMEM-compatible register layout 相关内容
- `Fixup.cpp` 中 WarpSpecialize / TMEM 相关内容

C500 对照关系：

| TLX/AMD/NVIDIA 概念 | C500/Gluon 当前处理 |
| --- | --- |
| `tlx-insert-require-layout` | 新建 `tritonmetaxgpu-gluon-insert-require-layout` |
| `tlx-propagate-layout` | 新建 `tritonmetaxgpu-gluon-propagate-layout` |
| TDM copy descriptor-compatible padded shared encoding | 不实现 TDM；改为 async_copy contiguous shared anchor |
| TMEM-compatible register layout | 不实现，记后续目标 |
| WarpSpecialize carrier propagation | 不实现，C500 当前无该语义 |
| storage alias | 不实现，记后续目标 |
| source-pinned layout | 不实现，记后续目标 |
| PrintTTGIRToTLX | 不实现，记后续目标 |

## 1. 新 pass 名字和顺序

最终 pipeline 要尽量长得像 TLX：

```text
Gluon inliner
tritonmetaxgpu-gluon-insert-require-layout
tritonmetaxgpu-gluon-propagate-layout
SCCP / CSE / canonicalizer
optimize_dot_operands
remove_layout_conversions
MetaX optimize_smem_usage
后续 backend passes
```

新 pass：

```text
tritonmetaxgpu-gluon-insert-require-layout
  文件: third_party/metax/lib/Gluon/GluonInsertRequireLayout.cpp
  对应 TLX: tlx-insert-require-layout
  责任: 只合成 gluon.require_layout 约束，不做传播，不做最终 cleanup。

tritonmetaxgpu-gluon-propagate-layout
  文件: third_party/metax/lib/Gluon/GluonPropagateLayout.cpp
  对应 TLX: tlx-propagate-layout
  责任: dataflow propagation、retag、region consensus、require/release cleanup、
       fallback convert、gluon slice lowering、post-condition verify。
```

旧 pass 处理方式：

- `tritonmetaxgpu-gluon-insert-layout-constraints`：从 pipeline 删除，本轮重构不保留旧实现路径。
- `tritonmetaxgpu-gluon-propagate-layouts`：从 pipeline 删除，本轮重构不保留旧实现路径。
- `tritonmetaxgpu-gluon-materialize-layouts`：从 pipeline 删除。TLX 的 cleanup/materialize/fallback 属于 `PropagateLayout.cpp`，C500 也照此合并。
- `GluonLayoutUtils.cpp`：不再承载主逻辑。最多保留极少量无状态 helper，且 helper 必须被新 pass 明确调用。

旧 pass 名兼容不属于本轮设计。若以后必须兼容外部脚本，应另起纯机械 alias 任务；alias 只能转发新 pass，不能包含旧 seed/infer/materialize 逻辑。

## 2. 新文件结构

建议直接按 TLX 拆分：

```text
third_party/metax/include/Gluon/Analysis/
  GluonLayoutPropagation.h        # 从 TLX LayoutPropagation.h 复制改名

third_party/metax/lib/Gluon/Analysis/
  GluonLayoutPropagation.cpp      # 从 TLX LayoutPropagation.cpp 复制改名

third_party/metax/lib/Gluon/
  GluonInsertRequireLayout.cpp    # 从 TLX InsertRequireLayout.cpp 复制改名
  GluonPropagateLayout.cpp        # 从 TLX PropagateLayout.cpp 复制改名
  GluonC500AsyncCopyLayout.cpp    # C500 async_copy anchor / contiguous shared policy
  GluonC500LayoutHelpers.cpp      # 小型 MetaX/C500 encoding helper
```

对应构建文件：

- `third_party/metax/lib/Gluon/CMakeLists.txt`
- `third_party/metax/include/TritonMETAXGPUTransforms/Passes.td`
- `third_party/metax/include/TritonMETAXGPUTransforms/Passes.h`
- `third_party/metax/backend/compiler.py`

## 3. 第一性原理

Layout 不是装饰性 annotation，而是硬件执行映射：

```text
register tensor layout:
  logical tile element -> lane/register/value position

shared memdesc layout:
  logical tile element -> shared-memory byte address / bank / vectorized write pattern
```

因此自动推导不是“选一个看起来能过 verifier 的 layout”，而是求解一组来自消费者的方程：

```text
L(tensor_value) = required register encoding
M(memdesc_value) = required shared encoding
```

约束来源：

1. `tt.dot`
   - A/B operand 要 `DotOperandEncodingAttr`。
   - local_load result 的 dot operand encoding 可以反推 shared memdesc encoding。

2. `ttg.async_copy_global_to_local`
   - `src` / `mask` / `other` 必须同 layout。
   - destination memdesc 必须满足 C500 连续 shared 写入。
   - 如果 destination 后续被 `local_load -> dot` 消费，还必须满足 dot-shared encoding。

3. `tt.load` / `tt.store`
   - 第一阶段不做成大而全泛化框架，但也不是随意 hint。
   - `tt.load` 的 ptr/mask/other/result 和 `tt.store` 的 ptr/mask/value 必须最终满足 verifier 和 coalescing 要求。
   - 优先级低于 dot operand 和 async_copy destination shared 约束。

4. `gluon.extract_slice` / `gluon.insert_slice`
   - 是 subview transform edge。
   - parent/sub layout 可能相同，也可能需要 transpose/reshape/slice-aware 变换。

5. `scf.for` / `scf.if`
   - 只有 incoming/yield/result layout 达成 consensus 才能 retag。
   - 冲突时 tensor 可以 fallback convert；memdesc 冲突必须诊断或保持 unknown，不能伪造 memdesc conversion。

## 4. IR contract

继续使用 Gluon 现有 IR：

- `gluon.require_layout`
- `gluon.release_layout`
- `gluon.no_verify_encoding`
- `gluon.auto_encoding`

但主流程要改成 TLX 风格：

- `insert-require-layout` 只插入显式 `gluon.require_layout`。
- `propagate-layout` 读取 `require_layout` / `release_layout` 做 dataflow。
- 不再依赖旧的 `set_auto_layout` seed 列表作为主机制。
- 前端仍可产生 `AutoEncodingAttr`；新流程的职责是在 `propagate-layout` 结束前由显式 require/release 约束、subview inference 和少量默认 layout policy 将其消除。
- 对完全无消费者约束的 AutoEncoding，只允许使用确定性的 C500 default layout 或报错，不能靠旧 seed 列表静默猜。

语义要求：

```text
tensor require_layout:
  如果 producer 可 retag，则传播吸收。
  如果不能 retag，则 cleanup 成 ttg.convert_layout。

memdesc require_layout:
  必须传播到 local_alloc / memdesc view / loop carrier。
  如果 propagation 后仍不能 identity-fold，直接报错。
  不允许用 ttg.convert_layout 伪造 memdesc conversion。

release_layout:
  阻断专家 layout 继续向下游传播。
  cleanup 时 identity-fold 或降为 ttg.convert_layout。

no_verify_encoding:
  只允许中间阶段存在。
  propagate-layout 结束前必须 unwrap。
```

## 5. `GluonInsertRequireLayout.cpp`

此文件应直接从 TLX `InsertRequireLayout.cpp` 复制并局部改名。

保留并适配的 TLX 结构：

- `DotRewriteState`
- `DotRewriteLattice`
- `DotRewriteBackward`
- `DotConsumerInfo`
- `DotConsumerState`
- `DotConsumerLattice`
- `DotConsumerBackward`
- `findMemDescRoot`
- `isFedByAnyMemDescUser`
- `computeSharedEncFromDotEnc`
- `applyRequireLayout`
- `materializeTensorRequireLayout`
- `materializeDotUserTensorConstraints`
- `insertRequireLayout`

删除或替换的 TLX 结构：

- `isFedByTDM` 删除。
- `chooseTDMBufEncoding` 删除。
- `anchorTDMRequireLayout` 删除。
- `materializeTDMConstraints` 删除。
- AMD CDNA4/GFX1250 TDM padded descriptor 分支删除。

新增 C500 结构：

```text
isFedByC500AsyncCopyProducer(memdesc)
chooseC500AsyncCopySharedEncoding(copyOp, dstMemDesc, dotConsumerInfo)
anchorC500AsyncCopyRequireLayout(copyOp)
materializeC500AsyncCopyConstraints(module, builder, solver)
verifyC500AsyncCopyContiguousSharedWrite(copyOp, chosenEncoding)
```

### 5.1 dot-fed local_load anchor

直接照 TLX：

1. `DotRewriteBackward` 从 `tt.dot` A/B operand 反向传播 `DotOperandEncodingAttr`。
2. 遇到 dot-fed `ttg.local_load`：
   - 对 local_load result/dot path 插入 tensor-side `gluon.require_layout`。
   - 对 local_load memdesc operand 插入 memdesc-side `gluon.require_layout`。
3. 如果 local_load result 有 incompatible user：
   - 不 retag local_load。
   - 只保留 tensor require，最终 fallback 成 `ttg.convert_layout`。
   - 输出 remark。

### 5.2 C500 async_copy anchor

把 TLX TDM anchor 的位置换成 C500 async_copy anchor。

对每个 `ttg.AsyncCopyGlobalToLocalOp`：

1. 读取 destination memdesc operand。
2. 通过 `DotConsumerBackward` 查 downstream `local_load -> dot` 是否有 dot consumer。
3. 若有 dot consumer：
   - 用 dot operand encoding 推导 shared encoding。
   - 同时检查该 shared encoding 是否满足 C500 async_copy 连续 shared 写入。
4. 若没有 dot consumer：
   - 选择 C500 默认 contiguous shared encoding。
5. 在 async_copy destination memdesc operand 前插入 `gluon.require_layout`。
6. 如果已经有等价 `require_layout`，保持幂等。
7. 如果已有 concrete shared encoding 与新 encoding 冲突：
   - strict mode 下报错。
   - permissive mode 也不能静默改成离散 shared 写入；只能保留原布局并输出明确诊断。

shared encoding 选择必须是“候选生成 + 验证”，不能是单点硬编码：

```text
candidate 1: dot-derived swizzled shared encoding
candidate 2: user/shared-order-preserving variant
candidate 3: C500 async-copy-default contiguous shared encoding

accept(candidate) iff
  compatible_with_local_load_to_dot(candidate) &&
  compatible_with_async_copy_contiguous_write(candidate)
```

如果没有候选同时满足 dot 和 async_copy，必须 fail。不能选择 dot-only encoding，也不能选择 async-only encoding 后让 dot 前补一串不可控 conversion。

### 5.3 async_copy src/mask/other contract

此 contract 也归 `insert-require-layout` 合成：

```text
L(src)   = E_async_src
L(mask)  = E_async_src
L(other) = E_async_src    # 当 other 是 ranked tensor 时
```

要求：

- `src` / `mask` / `other` 形状兼容且 encoding 完全相同。
- 如果 `mask` / `other` 来自 `gluon.extract_slice`，约束应推回 parent。
- 如果 follower 已经 concrete 且冲突，后续 `propagate-layout` 只能插最小 tensor convert，不能改变 destination memdesc 来迁就 follower。

### 5.4 C500 shared 连续写入判定

这是正确性硬约束，不是优化项。

必须验证：

- async_copy source tensor 的 logical order 与 destination shared encoding 的 innermost order 能形成连续写入。
- `contiguity` attr 与 dtype/shape/vector width 一致。
- 对 `extract_slice` / `ttg.extract_tensor` 产生的子 tile，`elemIdx` / `ctaIdx` 在 chosen order 下连续。
- 对双 buffer / multi-stage pipeline，所有 stage 的同一 buffer family 使用同一 shared encoding。
- 不能为了 dot layout 产生 per-element scattered shared writes。

当前 matmul 好 IR 的典型形态：

```text
A: tensor<32x128x!tt.ptr<f16>, #blocked>
   -> <32x128xf16, #shared, #smem, mutable>
   #shared order = [1, 0]

B: tensor<128x32x!tt.ptr<f16>, #blocked1>
   -> <128x32xf16, #shared1, #smem, mutable>
   #shared1 order = [0, 1]
```

这些只是调试基线，不应把形状硬编码进 pass。

### 5.5 自动推导和优化的边界

`insert-require-layout` / `propagate-layout` 只负责 legality：

- 消除 AutoEncoding。
- 让 dot operand、async_copy、memdesc view、region carrier 类型一致。
- 在不能 retag tensor producer 时插最小 `ttg.convert_layout`。
- 对 memdesc 冲突直接诊断。

layout optimization 放到后续 pass：

- 在多个合法候选中选更好 shared order/vector width。
- 减少 dot 前 convert。
- 减少 async_copy 前 mask/other convert。
- 减少 store path scratch 和 SMEM 压力。

这保持 TLX 思路：require/propgate 解决“必须是什么”，RLC/MetaX 优化解决“哪个合法选择更快”。

## 6. `GluonLayoutPropagation.*`

此部分直接复制 TLX `Analysis/LayoutPropagation.h/cpp`。

保留：

- `LayoutEncoding`
- `LayoutEncodingLattice`
- `LayoutBackwardPropagation`
- `LayoutForwardPropagation`
- `TensorLayout`
- `TensorLayoutLattice`
- `TensorBackwardPropagation`
- `meet` / `join` 中 conflict -> unknown 的语义
- `MemDescTransOp` 使用 dialect transpose inference
- `MemDescReshapeOp` 使用 op type inference
- `RegionBranchOpInterface` carrier 处理

删除：

- WarpSpecialize 专用逻辑。
- TMEMSubSlice / TMEMCopy / DummyTMEM 逻辑。
- TLX `LocalAliasOp` / storage alias 逻辑。
- NVIDIA-only TMEM-compatible tensor layout 逻辑。

新增或适配：

- `gluon::RequireLayoutOp` / `gluon::ReleaseLayoutOp`。
- `gluon.extract_slice` / `gluon.insert_slice` 的 tensor subview inference。
- C500 async_copy 的 memdesc anchor provenance，用于冲突诊断。
- `NoVerifyEncodingAttr` unwrap。
- `AutoEncodingAttr` final double-check。

关键要求：

- `memdesc_trans` 不能直接复制 encoding，必须 transpose-aware inference。
- `memdesc_reshape` 不能直接复制 encoding，必须调用 type inference。
- region consensus 只能在所有 predecessor 类型一致时 retag。
- tensor conflict 可以 fallback convert。
- memdesc conflict 不能 fallback 成假 conversion。

## 7. `GluonPropagateLayout.cpp`

此文件直接复制 TLX `PropagateLayout.cpp` 并适配。

保留的 pass 结构：

```text
runOnFuncOp(func):
  如果没有 require/release 且没有可折叠 local_alloc/load fallback，提前返回。
  load DeadCodeAnalysis
  load SparseConstantPropagation
  load LayoutBackwardPropagation
  load LayoutForwardPropagation
  load TensorBackwardPropagation
  initializeAndRun
  computeBlockedTensorValues
  rewrite memdesc values from lattice
  rewrite retaggable tensor producer values from lattice
  update tensor region branch types
  verify post-conditions

runOnOperation():
  对每个 FuncOp 执行 runOnFuncOp
  apply cleanup patterns greedily
```

保留的 cleanup pattern：

- `RequireLayoutPattern`
- `ReleaseLayoutPattern`
- `FoldRetaggedLocalAllocLoad`
- `FoldLocalAllocLoadFallback`

C500/Gluon 新增 cleanup：

- `LowerGluonExtractSlicePattern`
- `LowerGluonInsertSlicePattern`
- `VerifyNoAutoEncodingPattern` 或 pass-level final check
- `VerifyNoNoVerifyEncodingPattern` 或 pass-level final check
- `VerifyC500AsyncCopyLayoutPattern`

最终 `propagate-layout` 结束后必须满足：

- 没有 `gluon.require_layout` 残留，除非已经 signal failure。
- 没有 `gluon.release_layout` 残留。
- 没有 `gluon.auto_encoding`。
- 没有 `gluon.no_verify_encoding`。
- `ttg.async_copy_global_to_local` 的 ranked `src` / `mask` / `other` encoding 一致。
- async_copy destination memdesc shared encoding 满足连续写入。
- `tt.dot` operand layout 合法。

## 8. Passes.td 和 pipeline 修改

`third_party/metax/include/TritonMETAXGPUTransforms/Passes.td` 新增：

```tablegen
def TritonMETAXGPUGluonInsertRequireLayout
    : Pass<"tritonmetaxgpu-gluon-insert-require-layout", "mlir::ModuleOp"> {
  let summary = "Insert explicit Gluon layout constraints for dot and C500 async_copy paths";
}

def TritonMETAXGPUGluonPropagateLayout
    : Pass<"tritonmetaxgpu-gluon-propagate-layout", "mlir::ModuleOp"> {
  let summary = "Propagate explicit Gluon layout constraints using TLX-style dataflow";
}
```

`third_party/metax/backend/compiler.py` 改成：

```python
passes.gluon.add_inliner(pm)
if not skip_gluon_layout_passes:
    metax.passes.ttgpuir.add_tritonmetaxgpu_gluon_insert_require_layout(
        pm, capability, store_coalesce
    )
    metax.passes.ttgpuir.add_tritonmetaxgpu_gluon_propagate_layout(pm)
```

删除旧 pipeline 调用：

```python
add_tritonmetaxgpu_gluon_insert_layout_constraints(...)
add_tritonmetaxgpu_gluon_propagate_layouts(...)
add_tritonmetaxgpu_gluon_materialize_layouts(...)
```

## 9. 实施阶段清单

### R0. 清理旧入口

- [ ] 从 pipeline 移除旧三 pass。
- [ ] 新增 TLX-style 两 pass 名字。
- [ ] 旧 pass 文件直接删除或编译期排除；本轮不保留旧实现路径。
- [ ] `GluonLayoutUtils.cpp` 不再被新 pass 依赖主逻辑。

### R1. 复制 TLX LayoutPropagation

- [ ] 复制 `LayoutPropagation.h/cpp` 到 Gluon Analysis。
- [ ] namespace 改成 MetaX/Gluon。
- [ ] `tlx::RequireLayoutOp` -> `gluon::RequireLayoutOp`。
- [ ] `tlx::ReleaseLayoutOp` -> `gluon::ReleaseLayoutOp`。
- [ ] 删除 WarpSpecialize/TMEM/storage alias 分支。
- [ ] 保留 `memdesc_trans` / `memdesc_reshape` inference。
- [ ] 编译通过。

### R2. 复制 TLX InsertRequireLayout dot 路径

- [ ] 复制 `DotRewriteBackward`。
- [ ] 复制 dot-fed `local_load` 发现逻辑。
- [ ] 插 tensor-side `gluon.require_layout`。
- [ ] 插 memdesc-side `gluon.require_layout`。
- [ ] unsupported user 输出 remark 并 fallback tensor convert。
- [ ] dot-only matmul IR 中不再需要手写 layout。

### R3. 用 C500 async_copy anchor 替换 TDM anchor

- [ ] 删除 TDM anchor 逻辑。
- [ ] 新增 `DotConsumerBackward` dot-aware async_copy shared 选择。
- [ ] 新增 async_copy destination memdesc `require_layout`。
- [ ] 新增 C500 default contiguous shared encoding。
- [ ] 新增 shared encoding candidate list：dot-derived、user-order-preserving、C500 contiguous default。
- [ ] 每个 candidate 同时验证 dot-compatible 和 async-copy-contiguous。
- [ ] 新增 contiguous shared write verifier。
- [ ] 新增 src/mask/other same-layout constraint。
- [ ] 新增 conflict provenance 诊断。

### R4. 复制 TLX PropagateLayout

- [ ] 复制 `RequireLayoutPattern` / `ReleaseLayoutPattern`。
- [ ] 复制 tensor/memdesc lattice rewrite。
- [ ] 复制 region consensus。
- [ ] 复制 local_alloc/local_load fallback fold。
- [ ] 删除 WarpSpecialize/TMEM 检查。
- [ ] 加 final `AutoEncoding` / `NoVerifyEncoding` 检查。

### R5. Gluon slice/subview 语义

- [ ] `gluon.extract_slice` parent -> sub inference。
- [ ] `gluon.extract_slice` sub -> parent inference。
- [ ] `gluon.insert_slice` base/update/result inference。
- [ ] 不把所有 slice 边都强行 same-layout。
- [ ] 最终 lowering 到 `ttg.extract_tensor` / `ttg.insert_tensor`。

### R6. C500 async correctness tests

- [ ] async_copy src/mask/other 相同 layout。
- [ ] async_copy destination shared 连续写入。
- [ ] async_copy destination 后接 local_load/dot 时 shared encoding 与 dot 一致。
- [ ] async_copy destination 无 dot consumer 时选择 C500 contiguous default。
- [ ] async_copy 与 dot 冲突时报错，不 silent fallback。
- [ ] multi-stage async pipeline 中同 buffer family encoding 一致。

### R7. Region 和 memdesc tests

- [ ] `scf.for` loop-carried tensor consensus。
- [ ] `scf.for` loop-carried memdesc consensus。
- [ ] `scf.if` result/yield consensus。
- [ ] `memdesc_trans` transpose-aware propagation。
- [ ] `memdesc_reshape` type-inference propagation。
- [ ] memdesc conflict 诊断。

### R8. 删除旧物化路径

- [ ] 删除旧 `materializeMetaXGluonLayouts` 主流程。
- [ ] 不再单独跑 `gluon-materialize-layouts`。
- [ ] `propagate-layout` 内完成 cleanup/fallback/lowering。
- [ ] RLC 前 IR 不含 Gluon layout contract ops。

## 10. 精度错误时的 IR diff 流程

如果修改后算子精度不对，优先比较：

- `/datapool/kezengxiang/2026/09-maca-matmul_tn_1.ttgir.mlir`
- `/datapool/kezengxiang/2026/09-maca-matmul_tn_2.ttgir.mlir`

命令：

```bash
diff -u \
  /datapool/kezengxiang/2026/09-maca-matmul_tn_1.ttgir.mlir \
  /datapool/kezengxiang/2026/09-maca-matmul_tn_2.ttgir.mlir | sed -n '1,260p'

rg -n '#blocked|#shared|#mma|ttg.async_copy_global_to_local|ttg.local_load|ttg.local_store|tt.dot|ttg.extract_tensor|ttg.insert_tensor|convert_layout|memdesc_trans|memdesc_reshape' \
  /datapool/kezengxiang/2026/09-maca-matmul_tn_1.ttgir.mlir \
  /datapool/kezengxiang/2026/09-maca-matmul_tn_2.ttgir.mlir
```

重点看：

- `#shared` / `#shared1` 的 `vec`、`perPhase`、`maxPhase`、`order`。
- async_copy source tensor encoding 是否分别为期望的 `#blocked` / `#blocked1`。
- async_copy mask/other 是否和 src 同 layout。
- async_copy destination memdesc 是否和后续 `local_load` operand type 一致。
- `ttg.local_load` result 是否是 dot operand encoding。
- `#mma` / `#mma1` / `#blocked2` 差异是否改变 accumulator 或 store path。
- `ttg.extract_tensor` / `ttg.insert_tensor` 的 `ctaIdx` / `elemIdx` 是否导致 shared 离散写入。
- loop-carried block argument 和 `scf.yield` 类型是否一致。
- dot 前、async_copy 前是否出现新增且无法被 RLC 消掉的 `ttg.convert_layout`。

常见错误归因：

- shared encoding 满足 dot 但破坏 async_copy 连续写入。
- async_copy `src` / `mask` / `other` encoding 不一致。
- slice/insert_slice 被错误当成 same-layout，导致子 tile index 错。
- `memdesc_trans` / `memdesc_reshape` 裸复制 encoding。
- region carrier conflict 被强行 retag。
- residual `convert_layout` 被插在 shared-memory 边界附近，改变写入/读取路径。

## 11. 验收标准

功能验收：

- [ ] Gluon matmul 不手写核心 tensor layout 也能生成合法 TTGIR。
- [ ] TTGIR 中无 `gluon.auto_encoding`。
- [ ] TTGIR 中无 `gluon.no_verify_encoding`。
- [ ] TTGIR 中无 `gluon.require_layout` / `gluon.release_layout`。
- [ ] async_copy `src` / `mask` / `other` ranked tensor layout 一致。
- [ ] async_copy 写 shared memory 连续。
- [ ] dot operands layout 合法。
- [ ] RLC 后没有明显多余 dot/async_copy 前 convert。

精度验收：

- [ ] `09-maca-matmul_tn_1.py` 精度通过。
- [ ] `09-maca-matmul_tn_2.py` 精度通过。
- [ ] 若失败，按第 10 节先看 IR diff，而不是盲调 encoding 常量。

工程验收：

- [ ] 新代码结构与 TLX 两 pass + Analysis 对齐。
- [ ] C500 特殊点集中在 `GluonC500AsyncCopyLayout.cpp`。
- [ ] 不再把新逻辑塞回旧 `GluonLayoutUtils.cpp`。
- [ ] 新增 lit/pytest 覆盖 dot、async_copy、slice、region、memdesc_trans/reshape。

## 12. 后续目标

这些现在不实现，但接口不要堵死：

- C500 或后续架构的 TDM-like descriptor copy。
- TMEM-compatible register layout。
- WarpSpecialize carrier propagation。
- storage alias。
- source-pinned layout。
- PrintTTGIRToTLX。
- 更完整的 cost model/layout optimization。

## 13. 当前实现反思与后续 TODO

当前实现已经跑通 `09-maca-matmul_common.py`、`09-maca-matmul_tn_1.py`、`09-maca-matmul_tn_2.py`、`09-maca-matmul_tt.py` 的 accuracy，并且主路径已经从旧三 pass 切到 TLX-style 两 pass。但从 `/datapool/kezengxiang/2026/tlx/docs/commit-performance-frontend-layout-report.md` 的模型看，现在还不是“纯 TLX require_layout contract”形态，而是有意采用了 Gluon seed 和 TLX contract 的混合方案：

```text
Gluon AutoLayout / set_auto_layout:
  用于给 auto encoding graph 放求解 seed。
  seed 表达约束，不是真实 convert_layout。
  适合 dot accumulator、global load/store、slice parent/sub 这类 value-level 求解入口。

TLX require_layout:
  用于从硬件 consumer 或硬边界反推 layout contract。
  适合 dot operand、local_load memdesc、async_copy destination memdesc。
  tensor require 不能吸收时 fallback 成 ttg.convert_layout；
  memdesc require 不能吸收时必须诊断，不能伪造 memdesc conversion。
```

这个混合方案和 TLX 报告中的第一性原理并不冲突：二者都是 layout constraint，不是用户手写真实搬运。区别在于 seed 来源、pass 边界和失败模式不同。后续重构的目标不是机械消灭 `set_auto_layout`，而是把它和 `require_layout` 的职责边界固定下来，避免两套机制互相污染。

### 13.1 seed/contract 分层 TODO

- [x] 明确写入代码注释和设计文档：`set_auto_layout` 只用于 AutoEncoding value 的求解 seed，`require_layout` 只用于硬件 consumer/boundary contract。
- [x] dot accumulator、`tt.load` result、`tt.store` ptr/mask、async slice parent 等 value-level 约束继续允许用 `set_auto_layout`，但必须在 `propagate-layout` 结束前完全 cleanup。
- [x] dot operand、local_load memdesc、async_copy destination memdesc 等硬边界优先使用 `require_layout`。
- [x] 避免在 insert pass 中半改 verifier-sensitive op 的 operand/result 类型；这类 op 应先种 seed，再由 propagate pass 统一 retag/materialize。
- [x] 补一个最终验证：RLC 前 IR 中不得残留 `gluon.set_auto_layout`、`gluon.require_layout`、`gluon.release_layout`、`gluon.auto_encoding`、`gluon.no_verify_encoding`。

### 13.2 代码结构收敛 TODO

- [ ] 从 `GluonInsertRequireLayout.cpp` 拆出 `GluonC500MmaLayout.cpp`，承载 C500 MMA encoding、dot accumulator sub-encoding、dot operand order 推导。
- [ ] 从 `GluonInsertRequireLayout.cpp` 拆出 `GluonC500AsyncCopyLayout.cpp`，承载 async_copy shared candidate、contiguous write verifier、src/mask/other contract。当前已拆出 propagate-time follower/verify 和 shared-write verifier；insert-time shared candidate 仍留在 Insert pass，避免把 DotConsumer solver 拆散。
- [x] 从 `GluonInsertRequireLayout.cpp` / `GluonPropagateLayout.cpp` 拆出 `GluonC500LayoutHelpers.cpp`，承载 AxisInfo order、global memory blocked encoding、element/pointee bitwidth 等通用 helper。
- [x] 保持 pass skeleton 接近 TLX：Insert pass 负责发现约束，Propagate pass 负责传播、retag、cleanup、fallback、lowering、final verify。
- [x] 不重新引入旧 `GluonLayoutUtils.cpp` 的大杂烩模式；helper 必须无状态、边界清楚、由新 pass 显式调用。

### 13.3 C500 async_copy correctness TODO

- [x] 在现有 order/vector/bytes 检查之外，补 `gluon.extract_slice` / `ttg.extract_tensor` 的 `ctaIdx`、`elemIdx` 连续性验证。
- [ ] multi-stage async pipeline 中，同一个 memdesc buffer family 必须使用同一 shared encoding；冲突时报出 root buffer 和所有冲突来源。
- [ ] async_copy destination 后接 dot consumer 时，candidate 必须同时满足 dot-compatible 和 async-copy-contiguous；不能 dot-only，也不能 async-only。
- [ ] async_copy destination 无 dot consumer 时，C500 default contiguous shared encoding 的选择逻辑需要独立测试。
- [ ] async_copy `src` / `mask` / `other` 来自 slice 时，parent/sub layout 推导必须保持 follower 与 src 同构，不能按 mask `i1` bitwidth 重新计算 parent vector width。
- [ ] 冲突诊断增加 provenance：说明约束来自 dot、async_copy、existing require_layout、用户 concrete memdesc 还是 loop carrier。

### 13.4 global load/store policy TODO

- [ ] 将普通 `tt.load` 的 ptr/mask/other/result same-layout 约束抽象成 C500 global memory policy，而不是散落在 pass 主体里。
- [ ] 将普通 `tt.store` 的 ptr/mask memory-side seed 与 value materialize convert 明确分开，避免 store value 反向污染 MMA accumulator。
- [ ] AxisInfo 推导连续维优先级高于 shape heuristic；shape heuristic 只能作为 fallback。
- [ ] 后续可接入 CoalescedLayout 风格的 placeholder：Python 只表达 coalesced 意图，C500 pass 根据 AxisInfo 和 dtype 生成 concrete blocked layout。
- [ ] 不在 Insert pass 中直接改 `tt.load` / `tt.store` operand type，以免 verify_each 在 result/value 还未传播时失败。

### 13.5 subview 和 region TODO

- [ ] 为 `gluon.extract_slice` parent -> sub、sub -> parent 分别补 lit case，覆盖 blocked、MMA、dot operand layout。
- [ ] 为 `gluon.insert_slice` base/update/result 推导补 lit case，特别是 full MMA accumulator 和 sub-MMA tile。
- [ ] 对 `ttg.extract_tensor` / `ttg.insert_tensor` lowering 的 `ctaIdx` / `elemIdx` 做 IR diff 辅助检查，防止子 tile index 错误。
- [ ] `scf.for` loop-carried tensor consensus、memdesc consensus、yield/result 类型一致性需要专项测试。
- [ ] `scf.if` 两分支 yield/result consensus 需要专项测试。
- [ ] memdesc conflict 不能 fallback 成 tensor convert；必须诊断。

### 13.6 测试和验收 TODO

- [ ] 新增 dot-fed local_load memdesc require lit 测试。
- [ ] 新增 async_copy src/mask/other same layout lit 测试。
- [ ] 新增 async_copy contiguous shared write lit 测试。
- [ ] 新增 async_copy destination 后接 local_load/dot 的 shared encoding 测试。
- [ ] 新增 async_copy 与 dot shared encoding 冲突时报错测试。
- [ ] 新增 global `tt.load` / `tt.store` AutoEncoding 消解测试。
- [ ] 新增 `memdesc_trans` transpose-aware propagation 测试。
- [ ] 新增 `memdesc_reshape` type-inference propagation 测试。
- [ ] 保留算子回测：`python/tutorials/gluon/09-maca-matmul_common.py`、`09-maca-matmul_tn_1.py`、`09-maca-matmul_tn_2.py`、`09-maca-matmul_tt.py`。
- [ ] 若精度失败，优先按第 10 节看 IR diff，特别是 shared encoding、async_copy follower layout、dot operand layout、subview index。

### 13.7 optimization 后置 TODO

- [ ] 不把完整 cost model 塞进 insert/propagate pass；这两个 pass 先保证 legality 和硬件 correctness。
- [ ] layout optimization 后置给 RLC / remove-layout-conversions / MetaX optimize_smem_usage。
- [ ] 后续可做 shared layout candidate scoring，但必须在 legality candidate 集合内选择。
- [ ] 后续可做 SMEM budget-aware conversion elimination，避免为了高 vectorization 推爆 scratch。
- [ ] 后续可做 store-only conversion fold，剪掉 `local_store(reshape(convert_layout(x)))` 这类热路径冗余转换。
