# MetaX Gluon 扩展：显式细粒度流水与隐藏式 Layout 推导

## 1. 核心论点


用户只要显式写细粒度流水，不写 layout，降低用户心智。

- 用户显式描述细粒度流水：什么时候发 async copy，什么时候等 GVM，什么时候 barrier，什么时候 local load，什么时候做 B operand permute，什么时候 dot，什么时候更新 accumulator。
- 编译器隐藏layout：global、shared、register、dot operand、MMA accumulator、store-facing layout 之间的 physical encoding 由后端联动推导和物化。


## 2. 只显式写细粒度流水

MetaX Gluon 暴露给用户的是一套数据流式编程模型。用户关心的是流水如何推进，而不是每个 tensor 最终落成什么 physical layout。

核心前端接口如下：

| 前端表达 | 语义 |
|---|---|
| `gl.local_alloc` | 建立 shared memory staging buffer |
| `gl.metax.async_copy_global_to_shared` | 显式发起 global 到 shared 的异步搬运 |
| `gl.metax.gvm_arrive` | 标记 GVM copy 到达关系 （后期改为编译器自动推导） |
| `gl.metax.barrier` | 控制全局流水边界 （后期改为编译器自动推导） |
| `gl.metax.barrier_shared` | 控制 shared memory 可见性边界 （后期改为编译器自动推导） |
| `shared_descriptor.load(...)` | 从 shared 取 register 或 MMA operand |
| `gl.metax.bsm_perm` | 表达 B operand 进入 MACA MMA 前的重排 |
| `gl.dot` | 表达 MMA 计算 |
| `gl.metax.slice` | 从 parent tile 中取 micro tile 到硬件层lowering为 ttg.extract_tensor|
| `gl.metax.slice_update` | 把 micro tile 写回 parent   到硬件层lowering为 ttg.insert_tensor |
| `gl.metax.iglp` | 插入 MetaX 指令组调度 hint （后期改为编译器自动推导） |
| `gl.metax.sched_bound` | 插入调度边界 （后期改为编译器自动推导） |

在 `09-maca-matmul.py` 里，K-loop 的核心不是线性的 `load -> dot -> store`，而是把下一批数据搬运和当前批计算交织起来：

```text
prefetch next tile
  overlaps with
compute current tile
  synchronized by
gvm_arrive / barrier / barrier_shared
```

用户显式控制：

- shared buffer 有几级 staging。
- 每个 stage 什么时候发起 async copy。
- copy 和 compute 如何重叠。
- 什么时候等待 GVM 到达。
- 什么时候从 shared load 到 register。
- B operand 是否需要做 `bsm_perm`。
- 每个 micro tile 的 dot 顺序。
- accumulator 子块什么时候合回 parent accumulator。
- `iglp` / `sched_bound` 这种 MetaX 调度 hint 放在哪里。

这就是“显式细粒度流水”的含义：用户写的是 MACA matmul 的细粒度流水调度。

## 3. Layout 隐藏,后端联动推导

MetaX Gluon 的另一半是 layout 隐藏。用户显式写流水，但不直接写完整 physical layout。

原因很简单：这些 layout 不是独立局部决策，而是互相约束。

整体关系如下：

```text
global pointer tile
  -> global BlockedEncoding
  -> async_copy_global_to_shared
  -> shared memdesc
  -> SwizzledSharedEncoding
  -> local_load
  -> DotOperandEncoding
  -> dot
  -> MACAMmaEncoding accumulator
  -> slice / slice_update
  -> micro tile accumulator
  -> convert / store
  -> store-facing layout
```

各 layout 的约束来源如下：

| Layout | 主要约束来源 | 影响 |
|---|---|---|
| Global blocked layout | async copy source、mask、other | global coalescing 和 GVM 搬运形态 |
| Shared swizzled layout | local alloc、async copy destination、local load | shared bank conflict 和 operand load 形态 |
| Dot operand layout | local load 的 consumer 是 dot | A/B 如何进入 MMA |
| MACA MMA layout | dot、dtype、tile shape、warps、A/B order、store policy | accumulator 排布和 MMA 指令形态 |
| Slice parent/sub layout | slice、slice_update、dot micro tile | parent tile 和 micro tile 是否可合法映射 |
| Store-facing layout | store path、storeCoalesce | C 写回合并和 layout conversion 成本 |

- `dot` 需要的 A/B operand layout 依赖 MMA layout。
- MMA layout 依赖 dtype、tile shape、warps、A/B order、store policy（scenario="storeCoalesce"）。
- `async_copy` 需要 global blocked layout，但这个 layout 还要和 shared layout+local load 兼容。
- shared layout 不能只看 `local_alloc`，还要看后续 `local_load` 和 dot operand。
- `slice / slice_update` 的 parent/sub layout 必须等 concrete encoding 出来后才能判断是否合法。
- 前端过早写死 layout，容易引入多余 `convert_layout`，破坏 copy/compute overlap 和后端优化空间。

所以最终 layout 是多源约束收敛：

```text
async_copy constraint
+ local_load constraint
+ dot constraint
+ slice parent/sub constraint
+ store constraint
= final concrete TTG layout
```



## 4. `slice / slice_update` 如何推出 TTG physical slice

这一节按“解数学题”的方式讲，只用你给的例子：

```python
b_next_ptrs0_mask = gl.metax.slice(next_b_mask, [BLOCK_K, 32], [0, 0 * 32])
```

最终 materialize 出：

```mlir
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
%b_next_ptrs0_mask = "ttg.extract_tensor"(%next_b_mask)
  <{ctaIdx = array<i64: 0, 1>, elemIdx = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}>
  : (tensor<128x128xi1, #blocked2>) -> tensor<128x32xi1, #blocked1>
  loc(#loc298)
```

先说实话：**只看这一句 `gl.metax.slice(...)`，不能推出 `#blocked1/#blocked2`。**

这一句只能推出逻辑切片：

```text
source = next_b_mask
slice shape = [BLOCK_K, 32]
offsets = [0, 0 * 32]
```

`#blocked1/#blocked2` 是 layout propagation 的结果，依赖这个值后面作为 async-copy mask/source 的上下文。也就是说，完整题目不是只有 `slice`，而是：

```text
已知 1：前端逻辑 slice
已知 2：这个 slice result 被 async_copy_global_to_shared 当作 mask/source 使用
已知 3：MetaX async-copy memory-side layout policy
求：source/result concrete layout，以及 materialize 后的 ttg.extract_tensor indices
```

### 4.1 已知：前端逻辑 slice

在这个 kernel 中：

```text
BLOCK_K = 128
```

所以：

```python
gl.metax.slice(next_b_mask, [BLOCK_K, 32], [0, 0 * 32])
```

等价于：

```python
gl.metax.slice(next_b_mask, [128, 32], [0, 0])
```

因此前端给出的逻辑信息是：

```text
source shape = [128, 128]
result shape = [128, 32]
offsets      = [0, 0]
```

数学语义就是：

```text
b_next_ptrs0_mask[i, j] = next_b_mask[i + 0, j + 0]
0 <= i < 128
0 <= j < 32
```

也就是：

```text
b_next_ptrs0_mask = next_b_mask[0:128, 0:32]
```

这一步没有 layout，只有逻辑 shape 和 offsets。

### 4.2 什么时候生成 `#blocked1/#blocked2`

`#blocked1/#blocked2` 不是在创建 `ttg.extract_tensor` 时生成的。顺序是：

```text
1. insert-layout-constraints
   收集 async_copy / dot / load/store 等约束。
   非 dot AutoLayout 关系插入 set_auto_layout anchor。
   dot A/B tensor contract 插入 gluon.require_layout SSA wrapper。
   dot C/D accumulator contract 仍作为 AutoLayout seed。

2. propagate-layouts
   跑 Gluon layout inference，把 AutoLayout 解析成 concrete layout，
   并把 set_auto_layout / require_layout 都作为 seed 收集。
   #blocked1/#blocked2 在这里被写进 value type。

3. materialize-layouts
   这时 source/result 已经有 #blocked1/#blocked2。
   cleanup contract，补必要 conversion。
   rewriteGluonTensorGlueOps 只负责算 ctaIdx/elemIdx，并创建 ttg.extract_tensor。
```

所以时间顺序是：

```text
先有 #blocked1/#blocked2
后有 ttg.extract_tensor
```

### 4.3 如何生成 `#blocked1`

`b_next_ptrs0_mask` 是 async copy 的 mask。代码路径是：

```text
seedAsyncCopyBlocked(copyOp)
  -> seedAsyncCopyMemoryOperand(copyOp.getSrc(), ...)
  -> seedAsyncCopyMemoryOperandLike(copyOp.getMask(), ..., srcEncoding)
```

mask 要跟 async copy source 的 memory-side layout 对齐。因此 slice result 被 seed 成：

```text
b_next_ptrs0_mask : tensor<128x32xi1, #blocked1>
```

其中：

```text
#blocked1.sizePerThread = [8, 1]
#blocked1.threadsPerWarp = [16, 4]
#blocked1.warpsPerCTA = [1, 4]
#blocked1.order = [0, 1]
```

这里也要实事求是：`#blocked1` 不是从 `slice shape=[128,32]` 单独算出来的。它来自 async-copy memory operand 的 layout policy，再传给 mask。

### 4.4 如何生成 `#blocked2`

现在已经有：

```text
result layout = #blocked1
result shape  = [128, 32]
offsets       = [0, 0]
```

后端还需要给 parent：

```text
next_b_mask : tensor<128x128xi1, ?>
```

找一个 layout，使得：

```text
extract next_b_mask[0:128, 0:32]
```

能得到：

```text
tensor<128x32xi1, #blocked1>
```

代码路径是：

```text
inferAsyncCopyExtractParentEncoding
  -> inferExtractSourceEncodingWithMetaX / inferStoreCoalescedRepNParentEncoding
  -> tryExtractParentBlockedEncodingWithMetaX
  -> inferStableTTGExtractResultType
  -> ttg::ExtractTensorOp::inferReturnTypes
```

这一步的本质是“猜 parent layout，然后验证”：

```text
candidate parent layout
+ logical slice shape [128,32]
+ offsets [0,0]
=> 用 ExtractTensorOp::inferReturnTypes 推 result type
=> 如果 result layout 正好稳定为 #blocked1，则 candidate 成立
```

在这个例子里，成立的 parent layout 是：

```text
next_b_mask : tensor<128x128xi1, #blocked2>
```

其中：

```text
#blocked2.sizePerThread = [8, 4]
#blocked2.threadsPerWarp = [16, 4]
#blocked2.warpsPerCTA = [1, 4]
#blocked2.order = [0, 1]
```

这里不能说 `#blocked2` 是由 `128 / 32 = 4` 直接推出的。更准确地说：

```text
128 / 32 = 4 解释了为什么 parent dim1 可以看成 4 个宽度为 32 的 sub tiles；
但真正选择 #blocked2 的是 layout constraint + inferReturnTypes 稳定性验证。
```

### 4.5 数学求 `ctaIdx/elemIdx`

现在进入 materialize 阶段。此时已知：

```text
sourceTy = tensor<128x128xi1, #blocked2>
resultTy = tensor<128x32xi1,  #blocked1>
offsets  = [0, 0]
```

先求 logical sub-tensor index。代码是：

```text
logicalSubTensorIdx[d] = offsets[d] / resultShape[d]
```

所以：

```text
logicalSubTensorIdx[0] = 0 / 128 = 0
logicalSubTensorIdx[1] = 0 / 32  = 0

logicalSubTensorIdx = [0, 0]
```

接着调用：

```text
calExtractTensorIdx(sourceTy, resultTy, logicalSubTensorIdx)
```

#### 4.5.1 求 parent 的 shapePerCTA

代码里的公式是：

```text
shapePerCTA[d] = sizePerThread[d] * threadsPerWarp[d] * warpsPerCTA[d]
```

对 `#blocked2`：

```text
sizePerThread = [8, 4]
threadsPerWarp = [16, 4]
warpsPerCTA = [1, 4]
```

所以：

```text
shapePerCTA[0] = 8 * 16 * 1 = 128
shapePerCTA[1] = 4 * 4  * 4 = 64

shapePerCTA = [128, 64]
```

parent shape 是 `[128,128]`，所以：

```text
numCTAs[0] = 128 / 128 = 1
numCTAs[1] = 128 / 64  = 2

numCTAs = [1, 2]
```

#### 4.5.2 分维度求候选 cta/elem

维度 0：

```text
totalSizePerThread[0] = #blocked2.sizePerThread[0] = 8
subSizePerThread[0]   = #blocked1.sizePerThread[0] = 8
```

二者相等，代码走：

```text
extractSubCtaAndAllElem
```

这个函数做：

```text
subNumCtas = resultShape[0] / shapePerCTA[0]
           = 128 / 128
           = 1

cta_dim0 = logicalSubTensorIdx[0] * subNumCtas + [0 ... subNumCtas-1]
         = 0 * 1 + [0]
         = [0]

elem_dim0 = [0 ... totalSizePerThread[0]-1]
          = [0,1,2,3,4,5,6,7]
```

维度 1：

```text
totalSizePerThread[1] = #blocked2.sizePerThread[1] = 4
subSizePerThread[1]   = #blocked1.sizePerThread[1] = 1
```

二者不相等，代码走：

```text
extractAllCtaAndSubElem
```

这个函数做：

```text
cta_dim1 = [0 ... numCTAs[1]-1]
         = [0,1]

elem_dim1 = logicalSubTensorIdx[1] * subSizePerThread[1]
            + [0 ... subSizePerThread[1]-1]
          = 0 * 1 + [0]
          = [0]
```

此时：

```text
oneDimCtaIdx[0]  = [0]
oneDimCtaIdx[1]  = [0,1]

oneDimElemIdx[0] = [0,1,2,3,4,5,6,7]
oneDimElemIdx[1] = [0]
```

#### 4.5.3 线性化

`order = [0,1]`。代码调用：

```text
getSubLinearIndex(multiDimIndex, shape, order)
```

对 `order=[0,1]`，二维线性化就是：

```text
linear(i0, i1; shape0, shape1) = i0 + shape0 * i1
```

CTA 线性化：

```text
shape = numCTAs = [1,2]

linear(0,0) = 0 + 1 * 0 = 0
linear(0,1) = 0 + 1 * 1 = 1

ctaIdx = [0,1]
```

Elem 线性化：

```text
shape = totalSizePerThread = [8,4]

linear(0,0) = 0
linear(1,0) = 1
...
linear(7,0) = 7

elemIdx = [0,1,2,3,4,5,6,7]
```

### 4.6 得到 TTG op

因此 materialize 生成：

```mlir
%b_next_ptrs0_mask = "ttg.extract_tensor"(%next_b_mask)
  <{ctaIdx = array<i64: 0, 1>, elemIdx = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}>
  : (tensor<128x128xi1, #blocked2>) -> tensor<128x32xi1, #blocked1>
```

完整链路压缩成一道题就是：

```text
题目：
  gl.metax.slice(next_b_mask, [128,32], [0,0])
  result layout = #blocked1 [8,1]
  parent layout = #blocked2 [8,4]

求：
  ttg.extract_tensor 的 ctaIdx/elemIdx

解：
  logicalSubTensorIdx = [0/128, 0/32] = [0,0]
  shapePerCTA(parent) = [8*16*1, 4*4*4] = [128,64]
  numCTAs(parent) = [128/128, 128/64] = [1,2]

  dim0: SPT 相等 8==8
    cta=[0], elem=[0..7]

  dim1: SPT 不等 4!=1
    cta=[0,1], elem=[0]

  线性化 order=[0,1]
    ctaIdx=[0,1]
    elemIdx=[0,1,2,3,4,5,6,7]
```

这就是从前端直观的 logical slice 推到 TTG physical slice 的数学过程。

`slice_update` 到 `ttg.insert_tensor` 是同一套逻辑的反方向。它保存 base/update shape 和 offsets；layout propagation 得到 base/update/result layout；materialize 用同一个 `calExtractTensorIdx` 算 `ctaIdx/elemIdx`，然后生成：

```text
ttg.insert_tensor(base, update) {ctaIdx, elemIdx}
```

它的元素级语义是：

```text
result = base
result[offset_m + i, offset_n + j] = update[i, j]
```

因此 `slice / slice_update` 的真正作用是：

- 让前端能表达 micro tile 级 MMA 数据流。
- 让用户在流水里自然地切 A/B/acc 子块。
- 让写法贴近算法里的 tile/sub-tile 关系，比直接暴露 `ttg.extract_tensor / ttg.insert_tensor` 更直观。
- 保持 layout 隐藏，不提前绑定 physical encoding。
- 让后端在 layout 确定后，把逻辑切片翻译成 concrete layout 下的 TTG physical slice。

这也是为什么 `09-maca-matmul.py` 可以大量用 `slice` 切 A/B/acc 子块，同时仍然不要求用户手写最终 layout。

## 5. 后端三段式 layout pipeline

MetaX 后端没有继续使用通用的 `gluon.add_resolve_auto_encodings` 路径，而是引入了专门的三段式 layout pipeline。

核心 pass 顺序如下：

```text
tritonmetaxgpu-gluon-insert-layout-constraints
tritonmetaxgpu-gluon-propagate-layouts
tritonmetaxgpu-gluon-materialize-layouts
```

三者职责如下：

| Pass | 作用 |
|---|---|
| insert-layout-constraints | 从 dot、async copy、load/store 等数据流发现 layout 约束；非 dot 和 dot C/D 关系插 `gluon.set_auto_layout`，dot A/B tensor contract 插 `gluon.require_layout` wrapper |
| propagate-layouts | 沿 AutoLayout 图传播 layout，收集 `set_auto_layout` / `require_layout` seed，并处理 slice/update 的 parent-sub 关系 |
| materialize-layouts | 清理 layout contract，把 AutoLayout 收敛成 concrete TTG layout，降低 slice/update，插必要 conversion |

更细地看：

- `insert-layout-constraints` 从硬件语义强的 op 出发，例如 `dot`、`async_copy_global_to_shared`、`load/store`。当前这是一个 TLX 收敛过渡层：dot A/B 的 tensor contract 显式进入 `gluon.require_layout` wrapper，dot C/D accumulator 和其它多 operand 或 parent/sub 关系仍保留为 `gluon.set_auto_layout` seed。
- `propagate-layouts` 调用 Gluon layout inference，并使用 MetaX hooks 处理 `extract_slice / insert_slice` 的父子 layout 推导。它会同时收集 `gluon.set_auto_layout` 和 `gluon.require_layout` 作为 seed。
- `materialize-layouts` 在 concrete layout 已经确定后，把隐藏 layout 变成真实 TTG encoding，清理 `gluon.require_layout / gluon.release_layout`，并把 `gluon.extract_slice / gluon.insert_slice` 降成 TTG tensor op。

这里的 `gluon.require_layout` 还不是 TLX 的最终形态。当前它是 dot A/B tensor-side internal wrapper：

```text
显式记录 dot A/B tensor contract
参与 Gluon AutoLayout 推导
cleanup 时 identity 或 convert
```

它暂时还不会像 TLX `tlx.require_layout` 那样作为完整 SSA wrapper 去统一处理
memdesc producer retag、RegionBranch consensus 和全路径 legality/fallback。长期目标是
把这条路径逐步收敛到 TLX 的处理模式：

```text
constraint synthesis -> legality/dataflow -> producer/carrier retag -> residual convert cleanup
```

后续还有这些优化：

```text
optimize_dot_operands
remove_layout_conversions
optimize_smem_usage
reduce_data_duplication
reorder_instructions
```

这些 pass 的角色不是重新决定主 layout，而是清理和优化物化后的结果，降低 conversion、shared memory 和 dot operand 成本。

## 6. 用 `09-maca-matmul.py` 串完整执行流

`09-maca-matmul.py` 可以按下面这条数据流理解：

1. 构造 A/B/C tile offsets。
2. `gl.local_alloc` 建立 A/B shared staging buffer。
3. 预取多个 buffer slot 到 shared。
4. `gvm_arrive + barrier` 建立初始 copy/compute 边界。
5. `local_load` 从 shared 取 A/B operand。
6. `bsm_perm` 调整 B operand。
7. K-loop 内交织下一轮 async copy 和当前 MMA。
8. `slice(acc)` 拆出 micro accumulator。
9. `slice(a/b)` 拆出 MMA input micro tile。
10. `gl.dot` 更新 micro accumulator。
11. `slice_update(acc, c*)` 合回 parent accumulator。
12. loop 结束后转 dtype 并 store C。

这条执行流里，用户真正写的是：

```text
copy / wait / load / permute / dot / update
```

而没有写：

```text
global blocked layout
shared swizzled layout
dot operand layout
MMA accumulator layout
store-facing layout
```

这些 layout 由后端从数据流约束里推导出来。

## 7. 实现改动对应关系

这次扩展可以按层看：

| 层次 | 代表改动 | 作用 |
|---|---|---|
| Python API | `language/metax/__init__.py` | 暴露 MetaX 数据流 primitive |
| Python core | `_core.py`、`_semantic.py` | 支持 `local_alloc`、扩展 `shared.load`、增加 `gl.dot` |
| Layout 表达 | `_layouts.py` | 增加 `MACAMmaLayout` Python 表达 |
| Gluon IR | `GluonOps.td` | 增加 `gluon.extract_slice / gluon.insert_slice` |
| IR builder | `python/src/gluon_ir.cc` | 暴露 async copy、local load attrs、bsm perm、barrier、iglp、slice builder |
| Layout inference | `InferLayoutUtils.*` | 增加 layout hooks，支持 slice/update 传播 |
| MetaX passes | `GluonInsertLayoutConstraints.cpp`、`GluonPropagateLayouts.cpp`、`GluonMaterializeLayouts.cpp` | MetaX 专用 layout 插约束、传播、物化 |
| Backend pipeline | `third_party/metax/backend/compiler.py` | 接入三段式 Gluon layout pipeline |

从设计上看，Python API 负责让用户写细粒度流水，Gluon IR 负责保留 layout-parametric 意图，MetaX backend 负责把 layout 隐藏部分推导并落到 TTG。

## 8. 结论

MetaX Gluon 扩展的主轴是：

```text
用户只要显式写细粒度流水，不写 layout，降低用户心智。
```

这句话背后有两层含义：

- 用户显式控制 MACA kernel 的时间调度：async copy、GVM wait、barrier、shared load、permute、MMA、accumulator update。
- 编译器隐藏并联动推导空间布局：global、shared、register、dot operand、MMA accumulator、store-facing layout。

`slice / slice_update` 是这套设计里的关键中间层。它让前端能表达 micro tile 级别的数据流，但不提前绑定 physical layout；后端在 concrete layout 已经确定后，再把它们降成 `ttg.extract_tensor / ttg.insert_tensor`。

最终效果是：用户获得细粒度流水控制能力，同时不承担完整 layout 编排负担。
