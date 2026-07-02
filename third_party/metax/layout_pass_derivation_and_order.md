# metax 布局推导顺序与布局算法（含关键代码）

## 1. make_ttgir 中的 pass 顺序（默认路径）

在 `backend/compiler.py` 的 `make_ttgir` 里，metax 相关 pass 的顺序如下（按调用顺序）：

```python
# ll.253 起
if opt.pipeline == "cpasync" or opt.pipeline == "cpasync-mixed":
    metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_for_int8_pass(pm, opt.num_stages, opt.pipeline)
metax.passes.ttgpuir.add_accelerate_matmul(pm, opt.num_stages, disable_prefetch, store_coalesce, capability)
passes.ttgpuir.add_remove_layout_conversions(pm)
if not os.getenv("TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT"):
    metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_for_constancy_load_layout(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
if store_coalesce:
    metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_from_repn_to_elemn_pass(pm)
    metax.passes.ttgpuir.add_tritonmetaxgpu_optimize_cstore_pass(pm, opt.num_stages)
    passes.ttgpuir.add_remove_layout_conversions(pm)
...
if capability // 10 >= 80:
    if opt.pipeline == "basic":
        metax.passes.ttgpuir.add_pipeline_maca(pm, opt.num_stages, opt.pipeline_load_num, fullstage, False)
    else:
        metax.passes.ttgpuir.add_tritonmetaxgpu_addptr_opt_pass(pm, opt.num_stages, fullstage, mixed)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        metax.passes.ttgpuir.add_pipeline_async_tn(pm, opt.num_stages, inner_stages[0], inner_stages[1])
        metax.passes.ttgpuir.add_pipeline_async_tt(pm, opt.num_stages)
        metax.passes.ttgpuir.add_pipeline_async_base(pm, opt.num_stages, fullstage, mixed)
...
metax.passes.ttgpuir.add_tritonmetaxgpu_optimize_smem_usage(pm, reduce_smem_usage)
```

结论：**第一层是 `accelerate_matmul`，后面是若干布局修正 pass，再进 pipeline。**

---

## 2. `Accelerate` 是核心布局“推导”起点

`add_accelerate_matmul` 对应 `TritonMETAXGPUAccelerateMatmul`，在 `AccelerateMETAXMatmul.cpp`。

`BlockedToMMA`（核心 rewrite）从 `dot` 反推 MMA 布局：

```cpp
if (versionMajor == 2) {
  auto elemsPerThread = getDefaultElemsPerThread(elementTy, enableTf32, computeCapability);
  auto warpsPerTile = warpsPerTileMACA(dotOp, retShape, numWarps);
  if (enableOptMMA && aCvtOp && bCvtOp) {
     auto aorder = getOrder(aCvtOp);
     auto border = getOrder(bCvtOp);
     int version = -1;
     bool isOpt = updateLayout(elemsPerThread, warpsPerTile, tile, version,
                              numWarps, enableTf32, elemType,
                              aorder, border,
                              disablePrefetch, false, storeCoalesce,
                              computeCapability);
     if (isOpt) setAttr("use.opt.maca.mma", 1);
  }
  mmaEnc = MACAMmaEncodingAttr::get(..., warpsPerTile, elemsPerThread,...);
  // 创建新的 dot operand cvt + dot + final cvt
}
```

> 这一步决定 `dotOp` 的 MMA 编码（`MACAMmaEncodingAttr`）和 A/B 的 `DotOperandEncoding`。

---

## 3. `updateLayout` / `matchTable` 的推导算法

在 `MACACommon.h/cpp` 里定义：

```cpp
bool updateLayout(...,
                  llvm::SmallVector<unsigned, 3> &elemsPerThread,
                  llvm::SmallVector<unsigned, 2> &warpsPerTile,
                  llvm::SmallVector<int, 4> tile,
                  int &version, int numWarps, bool enableTf32,
                  Type dtype, ArrayRef<unsigned> aorder,
                  ArrayRef<unsigned> border,
                  bool disablePrefetch,...)
```

关键步骤：

1. 根据 `aorder/border` 查 `layouttable` 得出 `TN/NT/NN/TT`。
2. 如果允许 prefetch 且非 `chainDot`，`tile[2] /= 2`。
3. 按 dtype 从对应静态表匹配（`fp32table/tf32table/fp16table/fp16table_86/i8table/i8table_86`）：

```cpp
if (dtype.isF32() && !enableTf32)   isOpt = matchTable(pattern, fp32table,...)
else if (dtype.isF16()...)          isOpt = matchTable(pattern, ...)
else if (dtype.isInteger(8))         isOpt = matchTable(pattern, ...)
```

`matchTable` 的关键逻辑：

```cpp
if (table.count(pattern)) {
  auto newLayout = table.at(pattern)[0];
  if (storeCoalesce && table.at(pattern).size() > 1) newLayout = table.at(pattern)[1];
  auto newWarps = std::get<1>(newLayout);
  if (numWarps == newWarps[0] * newWarps[1]) {
    elemsPerThread = std::get<0>(newLayout);
    warpsPerTile = newWarps;
    return true;
  }
}
return false;
```

所以：**更新成功即返回可行的 `elemsPerThread + warpsPerTile`，`dot` 才重写成 MMA 布局。**

---

## 4. 后续布局 pass 的先后关系（真正的“推导链”）

### 4.1 `change_layout_for_constancy_load`（常量性布局）
文件：`ChangeLayoutForConstancyLoad.cpp`

1. 找到 `forOp` 里所有 `ConvertLayoutOp` 产出的 `dot` operand。
2. 从每个 dot backward slice 收集它对应 `loadOp`。
3. 对每个 `load`：若 `constancy>1 && contiguity(order[0])==1`，执行：
   - `getDivContiguityInterConstGroup(ptr, dim, axisInfo)` 递归分析 div/addptr/broadcast。
   - `calSizePerThreadWithConstancy()` 计算可提高的 `sizePerThread`。
4. 按 `(shape, order)` 取最大 `sizePerThread`，统一替换同组 load。

```cpp
int calSizePerThreadWithConstancy(load, axisInfo){
  best = min(128/elemBits, contiguityInterConstGroup[order0]);
  sizePerThread = min(best * constancy,
                      min(totalElems/threads, shapePerCTA[order0]));
  loadOp.setContiguityInterConstGroupAttr(...);
}
```

替换逻辑：

```cpp
auto newPtrType = RankedTensorType::get(oldPtrShape, oldPtrElemTy, newBlockedLayout);
auto newLoadOp = builder.create<LoadOp>(..., newPtr, Value(), Value(), ...);
auto newLoadCvtRes = builder.create<ConvertLayoutOp>(..., tensorType, newLoadOp);
targetLoad.getResult().replaceAllUsesWith(newLoadCvtRes);
```

### 4.2 `change_layout_for_int8`（仅 cpasync 且 num_stages==4）
文件：`ChangeLayoutForInt8.cpp`

- 前置条件：`numStages==4 && pipeline=="cpasync"`，且单 dot。
- 仅 NT 场景、仅 int8。
- 要求目标 layout 的 A/B order 构成 `TN`。
- 重写每个 load 的 `BlockedEncodingAttr`：`threadsPerWarp=[8,8]`, `sizePerThread[order0]=8`。
- 重建 load、ptr/mask/other 的 convertlayout，并保留 trans 节点。

```cpp
threadsPerWarp[0]=8; threadsPerWarp[1]=8;
sizePerThread[order[0]] = 8;
auto newBlockedLayout = BlockedEncodingAttr::get(...);
auto newLoadOp = builder.create<LoadOp>(newResType, newPtr, newMsk, newOther,...);
```

### 4.3 `change_layout_from_repn_to_elemn`（store_coalesce=true）
文件：`ChangeLayoutFromRepNToElemN.cpp`

- 条件：单 dot，且有 `MACAMma`。
- 若目标 B `load` 在原 layout 中 `sizePerThread` 在 N 维可被 `tN` 整除且 `originRepN % tN==0`，则
  `sizePerThread[order[1]] = tN`（从 `[1, tk]` 变 `[tN, tk]`）。
- 同样重建 load + convertlayout，保留 trans。

### 4.4 `change_transop_graph`
文件：`ChangeTransOpGraph.cpp`

- 将 `dotA/B` 前面的 `load -> cvt -> trans -> dot` 重构为显式共享内存节点：
  - `local_alloc`（swizzled shared）
  - `memdesc_trans`
  - `local_load`
- 这样后续 pipeline 能更容易跟踪共享路径。

```cpp
auto localAlloc = builder.create<LocalAllocOp>(..., MemDescType(swizzled), load.getResult());
auto newTrans = builder.create<MemDescTransOp>(localAlloc, transOp.getOrder());
auto localLoad = builder.create<LocalLoadOp>(dotDstType, newTrans);
transOp.getResult().replaceAllUsesWith(localLoad.getResult());
```

### 4.5 `optimize_cstore`
文件：`OptimizeCStorePass.cpp` 与 `MACACommon.cpp::OptimizeCStore`

- 只在 `numStages==1` 生效。
- 从 loop 结果 -> dot -> store 的单路径触发。
- 把 store 前链路中的 `fp/mul/convertlayout` 重写到 MMA 布局，减少共享内存回退写路径。
- 核心分支：当累加位宽可达标 (`bitwidth >= 32`) 时，尝试按 mma layout 直接重建 mul/cvt/store。

### 4.6 `optimize_smem_usage`
文件：`OptimizeSmemUsage.cpp` + `ConvertLayoutOpToLLVM.cpp`

- 若 `reduce_smem_usage` 开启，逐 `ConvertLayoutOp` 找到 `src=MACA mma` 且 `dst=Blocked`，加 `AttrSharedMemForceNoVec`。
- 该属性在 LLVM 降级时触发专用 swizzle 版本，走“禁止向量化”的共享内存路径：

```cpp
if (!op->hasAttr(AttrSharedMemForceNoVec)) return failure();
if (op->hasAttr(AttrSharedMemForceNoVec)) { ... transferWithinBlockSwizzling(...) ... }
```

---

## 5. Pipeline pass 如何依赖前序布局

`PipelineMACABasic.cpp` 中最关键的几步：

1. `collectOps` 收集候选 load（rank>=2，按规则）
2. `checkOpUses` 只保留一用链且最终是 `DotOperand` 的 load，构建 `loadsMapping`。

```cpp
if (auto convertLayout = dyn_cast<ConvertLayoutOp>(use)) {
  if (auto tensorType = dyn_cast<RankedTensorType>(convertLayout.getResult().getType()))
    if (isa<DotOperandEncodingAttr>(tensorType.getEncoding())) {
      isCandidate = true;
      loadsMapping[loadOp] = convertLayout;
    }
}
```

3. `createBufferTypes` 根据 `loadsMapping` 中的 `cvt` 源 type 和 dot operand 编码，构建 `MemDescType` 的 swizzled shared buffer。
4. pipeline emit 阶段按 `validLoads` / `validCvts` 重建 async/scratch 路径。

也就是：**前序 pass 先把数据切成“可被 dotoperand/cvt 识别”的布局，这些结果直接喂给 pipeline。**

---

## 6. 旁路（当前实现是空 pass）

`AddPtrOpt`、`PipelineAsyncBase/TN/TT` 在当前版本 `runOnOperation()` 为 `return;`（占位）。

---

## 7. 你可以复用的“推导顺序口诀”

- **先推导：** `accelerate_matmul` 推导 dot 的 MMA 主布局。
- **再修正：** 常量性/ int8/repN->elemN/图变换，把 input layout 对齐 pipeline 与存储路径。
- **再降级：** `optimize_*` 调整 convertlayout/store/smem 成本（不是再选主布局，而是微调代价和合法性）。
- **最后：** pipeline 读取这些布局证据，构建 `swizzled shared` buffer 并改写循环调度。

## 8. 可以否从 `09-maca-matmul.py` 直接推导 layout？

可以，且结论如下：

1. 这类结论是**可推导的**：脚本里显式写死了 `gl.BlockedLayout`、`gl.MACAMmaLayout`、`gl.DotOperandLayout`、`gl.SwizzledSharedLayout`，不是“黑箱自动生成”。
2. 该 gluon Kernel 的显式链路是：
   - `a_offsets/b_offsets` 使用 `layout=gl.blocked` 的 `SliceLayout`（源/掩码/步长张量都按 `blocked`）。
   - `a_operand_layout = DotOperandLayout(0, mma_layout, 0)`，`b_operand_layout = DotOperandLayout(1, mma_layout, 0)`。
   - `acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)`。
   - `shared_a_layout = SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1,0])`
     `shared_b_layout = SwizzledSharedLayout(vec=4, per_phase=8, max_phase=1, order=[1,0])`
     并通过 `allocate_shared_memory` 创建 `a_smem`/`b_smem`。
   - `a_smem.index(i).load(..., a_operand_layout, intrinsic=True, is_constant_offs=True)`
   - `b_smem.index(i).load(..., b_operand_layout, dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)`
   - 退出前 `gl.convert_layout(acc.to(...), blocked)` 再 `gl.store(...)`，所以最终写回 C 是 `blocked` 布局。

3. 但若你问“最终在 LLVM/backend 里最终落地的 layout 是否唯一”，这里要分层说：
   - 编译前端层（Gluon → TTIR）：上面这些 layout 是可静态恢复的。
   - 编译后端层（`make_ttgir` 的 gluon 路径）：会额外做 `optimize_dot_operands` / `remove_layout_conversions` / `optimize_smem_usage` 等清洗和微调，通常不再重算你这套主 layout，只做约束验证和代价相关调整。
   - 所以**能推导“主布局骨架”，但低层 `convert_layout` 的最终物化方式（如某些 swizzle/vectorized 细节）需看生成的 IR/汇编确认**。

## 9. “无 layout 时”的推导图（含可执行逻辑）

> 说明：这里的“无 layout”指 kernel 里 `dot`/`load`/`store` 没有显式 `layout` 语义，依赖 Triton+metax 默认布局和 pass 自动推导。

```mermaid
flowchart TD
    A[Python Gluon / Triton Kernel\n未显式写入布局] --> B[make_ttgir / gluon_to_ttgir]

    B --> C[resolve_auto_encodings\n给未标注 Tensor 注入缺省 blocked/ordered 编码]

    C --> D[optimize_dot_operands]
    D --> D1{识别 dot(A,B,C)\n返回元信息\n(M, N, K, dtype, M/N order)}
    D1 --> D2[collect operand types/shapes\nA/B ptr shape, order, contiguity]
    D2 --> D3[pass to AccelerateMETAXMatmul]

    D3 --> E[BlockedToMMA::rewriteDot]
    E --> F{是否命中 A/B operand 订单\n(TN/NT/NN/TT) 与 dtype table}
    F -->|否| F0[fallback: 保持原始 blocked / 无法进入 MMA 路径]
    F -->|是| G[构造 pattern= (orderA, orderB, tile, warp/elements hint)]

    G --> H{updateLayout(maybe prefetch缩放)}
    H --> H1[matchTable(table, pattern)]
    H1 --> H2{numWarps 匹配\n(table warps == numWarps)}
    H2 -->|否| F0
    H2 -->|是| I[返回 elemsPerThread + warpsPerTile + version]

    I --> J[创建 DotOperandEncoding + MACAMmaEncoding\n并插入 A/B convert/layout rewrite]
    J --> K[dot op 重写为 MMA version]

    K --> L[后续布局修正 pass 链]
    L --> L1[change_layout_for_constancy_load: 提升 sizePerThread（若 constancy>1）]
    L --> L2[change_layout_for_int8: cpasync/int8 专用 threadsPerWarp 重写]
    L --> L3[change_layout_from_repn_to_elemn: repN -> elemN]
    L --> L4[change_transop_graph: trans/load 重构为 local_alloc/memdesc_trans]

    L --> M[优化 pass]
    M --> M1[optimize_cstore: C 侧路径重整（num_stages=1）]
    M --> M2[optimize_smem_usage: 标记 SharedMemForceNoVec（可选）]

    M --> N[Pipeline pass 收集布局证据\ncheckOpUses/loadsMapping]
    N --> O[build swizzled shared MemDescType and emit MACA pipeline]
    O --> P[ConvertLayoutOpToLLVM 下沉\n根据属性决定 1D/2D swizzle 与向量化策略]
    P --> Q[PTX/汇编：最终 shared layout / registers 形态]

    F0 --> R[退化路径: 可能保留普通 dot/load\n或无法触发 MMA 优化]
    R --> Q
```

对应可提炼算法（伪代码）：

```cpp
layout_infer_auto_dot(KernelIR):
  dot_infos = collect_dot_ops(k)
  for each dot in dot_infos:
    if not explicit_layout(dot) and can_apply_optmmaa(dot):
      aorder, border = infer_operand_orders(dot)
      pattern = mk_pattern(tile, aorder, border)
      if dot.numWarps != 0 and env.disablePrefetch:
        pattern.modify_for_prefetch()

      table = select_table(dot.dtype, dot.enableTf32, cap)
      cand = matchTable(pattern, table, storeCoalesce)
      if cand.found && cand.warpsProduct == dot.numWarps:
        set_layout(dot, MACAMmaEncoding(cand.ept, cand.warpsPerTile))
        rewrite_load_cvt_dot_cvt(dot, cand)
      else:
        mark_fallback_to_blocked(dot)

  run_layout_corrections(dot)
  finalize = run_pipeline_and_lowering()
  return finalize
```

如果你只看“是否能在无 layout 时恢复”：

- A/B/C 的主布局是可以恢复的，核心依据是：`matchTable` + `table` + `numWarps` + dtype。
- 共享内存阶段的最终 swizzle/向量化/bank-aware 细节是 pass 与目标代码生成联合决定，通常**非唯一**，需要看落盘 IR/LLVM。

## 10. `blocked` 的默认值：都有啥？如何推导？

`BlockedEncodingAttr` 的关键字段是：

- `sizePerThread`
- `threadsPerWarp`
- `warpsPerCTA`
- `order`
- 可选：`CTALayout`

### 10.1 TritonGPU conversion 的默认（从无 encoding tensor 推出）

在 `lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp` 里，type converter 对没有 encoding 的 tensor 自动补默认：

```cpp
triton::gpu::BlockedEncodingAttr encoding =
    getDefaultBlockedEncoding(context, shape, numWarps, threadsPerWarp, numCTAs);
return tensorType.cloneWithEncoding(encoding);
```

`getDefaultBlockedEncoding` 在 `lib/Dialect/TritonGPU/IR/Dialect.cpp`：

```cpp
int rank = shape.size();
order = reverse(0..rank-1);  // [rank-1, ..., 0]
sizePerThread = [1, 1, ..., 1];
encoding = BlockedEncodingAttr::get(context, shape, sizePerThread,
                                  order, numWarps, threadsPerWarp, numCTAs);
```

再到 `BlockedEncodingAttr::get(shape, sizePerThread, order, numWarps, threadsPerWarp, numCTAs)` 内部（定义在 `TritonGPUAttrDefs.td`）按 order 做贪心分配：

- 先走 `rank-1` 个维度：
  - `threadsPerWarp[i] = clamp(threadsPerCTA, 1, remainingLanes)`
  - `warpsPerCTA[i] = clamp(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps)`
  - 同时更新剩余的 `remainingThreads / remainingLanes / remainingWarps`
- 最后一个 order 维度用剩余量补齐。

直观默认骨架（上层）：
`sizePerThread` 全 1，`order` 反向。
`threadsPerWarp/warpsPerCTA` 的每维具体值由上面算法和 `shape/numWarps/threadsPerWarp/numCTAs` 决定。

### 10.2 gluon 翻译层的默认 helper

在 `python/triton/tools/triton_to_gluon_translater/translator_helpers.py` 的 `default_blocked_layout`：

```python
size_per_thread = [1 for _ in range(rank)]
threads_per_warp = [1 for _ in range(rank)]
threads_per_warp[rank - 1] = get_num_threads_per_warp()  # 32
warps_per_cta = [1 for _ in range(rank)]
warps_per_cta[0] = num_warps
order = [i for i in range(rank - 1, -1, -1)]
```

这就是常见的前端 `no layout` 默认构造。
