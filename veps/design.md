# TileScript 设计方案 v3.0
## 面向 AI 辅助优化的跨硬件 Tile 编程语言

---

## 第零部分：一页纸摘要

**一句话定位**：TileScript 是一个把"优化决策权"从编译器启发式转移到"人 + AI agent"手中的 Tile 级编程语言。编译器的职责不是替你做决策，而是**忠实执行显式意图 + 输出结构化诊断**。

**三个核心假设（全部必须被验证，不能只被主张）**：

| 假设 | 内容 | 验证方式 | 验证时间 |
|------|------|----------|----------|
| H1 | AI agent 在拥有结构化诊断 + 显式控制的前提下，能在有限迭代内达到或超越编译器启发式优化 | 阶段0对照实验（见 §1.3） | 前 6 周 |
| H2 | 一套 Tile 级 IR 能同时 lower 到 SIMT（NVIDIA）和 DMA+Cube（Ascend）且性能损失可控 | M1 Go/No-Go 验证 | 前 4 周 |
| H3 | 显式控制所需的表达能力（Warp 特化、Pipeline、MMA 形状）无法在 Triton 现有架构内以插件形式补齐 | 阶段0中同步给出书面论证 + 实验反证 | 前 6 周 |

**如果 H1 不成立，项目应终止**——不是转向，是终止。因为 H1 是项目存在的理由，没有它 TileScript 只是"另一个性能不如 Triton 的 Tile 语言"。

**1.0 交付范围（10 人 × 10 月）**：
- NVIDIA 后端：MatMul / FlashAttention / LayerNorm / Softmax / Transpose，裸性能 ≥ Native 65%，AI 辅助后 ≥ 80%（有对照实验数据支撑）
- 结构化诊断系统（JSON Schema 固定，向后兼容）
- PyTorch 集成（CUDA 路径）
- Ascend 后端：**实验级**（vector_add / MatMul / LayerNorm，不含 FlashAttention），前提是 H2 通过

**明确不做的**：自动 Swizzle、自动 Pipeline、AutoTuner 产品化、AMD 后端、Ascend FlashAttention。

---

## 第一部分：问题定义与假设验证

### 1.1 现有工具的结构性局限（精确表述）

原方案对 Triton/TileLang 的批评偏泛，这里收窄为三条**可被反驳的具体声明**：

**声明 A：优化决策不可见**。Triton 的 `tl.dot` 到 MMA 的映射、Shared Memory layout 选择、Pipeline 阶段数（`num_stages`只是提示），都由编译器内部 pass 决定。用户拿到的是最终 PTX，没有"为什么选择这个 layout"的中间产物。AI agent 调优时只能修改 `BLOCK_M/BLOCK_N/num_warps/num_stages` 等少数外部参数——这是**参数搜索**，不是**优化**。

**声明 B：意图不可表达**。FlashAttention-3 的 producer/consumer warp 特化、Persistent Kernel 的 tile scheduler、跨 kernel 的 epilogue 融合，Triton 语言层面没有对应原语。TileLang 有部分能力但由编译器自动注入，用户不可控。

**声明 C：编译不确定**。Triton 同一份源码在不同版本、甚至不同 autotune 缓存状态下产生不同 PTX，AI 的 A/B 试验无法复现。

这三条中，B 和 C 是架构性的（Triton 修不了）；A 是可以通过插件缓解的。这正是 H3 需要论证的边界。

### 1.2 为什么不做 Triton 插件（对 Review 第七点的正面回应）

**能用插件做的**：把 `ptxas -v`、Nsight Compute 的 bank conflict / occupancy / stall reason 数据，结构化成 JSON 喂给 AI。这个工作量约 1 人月，**我们应该先做，并把它作为 H1 的验证工具**。

**插件做不到的**：
- Triton 的 `tl.dot` 无法指定 MMA 形状、无法指定 A/B 操作数在寄存器中的 fragment 布局
- 无法把两个 warp group 绑定到不同角色（producer 只做 TMA/cp.async，consumer 只做 MMA）
- 无法阻止 Triton 自己的 pipeline pass 重排你手写的 `cp.async` 顺序
- 诊断信息无法映射回源码行——因为 Triton IR 已经把用户代码结构打散

所以 TileScript 的存在理由不是"诊断"，而是**"诊断 + 可以按诊断修改的手柄"**。前者插件能做，后者不能。阶段0的对照实验必须同时测这两半：只有诊断（Triton+插件）能提升多少？诊断+手柄（TileScript 原型）能再提升多少？如果第二个增量不显著，H3 不成立，项目应改为 Triton 诊断插件。

### 1.3 阶段0：核心假设验证实验（前 6 周，3 人）

这是原方案完全缺失的部分，也是新方案最重要的增补。

**实验设计**：

```
被测对象：GEMM 4096³ FP16、FlashAttention seq=2048 d=64，H200
对照组 C0：Triton 官方 autotune 结果（编译器启发式基线）
对照组 C1：Triton + 诊断插件，AI agent（Claude/GPT 类）迭代 20 轮调外部参数
实验组 E1：TileScript 手写原型（不经编译器，直接手写 CUDA 模拟"显式控制"）
           + 同样的诊断 JSON，AI agent 迭代 20 轮修改 layout/pipeline/warp 分工
实验组 E2：同 E1，但去掉诊断，AI 只看 wall-clock 时间（消融：诊断本身的价值）

指标：
  - 20 轮后达到 Native 库性能的百分比
  - 达到 Native 70% 所需的迭代轮数（收敛速度）
  - AI 做出的修改中"有效修改"（性能提升 > 2%）的比例
```

**Go 判据**：E1 相对 C1 的最终性能提升 ≥ 8 个百分点，且 E1 相对 E2 的收敛速度提升 ≥ 2×。两个条件同时满足才认为 H1 + H3 成立。

**No-Go 处置**：
- E1 ≈ C1（手柄没用）：改做 Triton 诊断插件，团队缩减到 3 人，剩余人力释放
- E1 ≈ E2（诊断没用，手柄有用）：TileScript 退化为"带显式控制的 CuTe Python 前端"，仍可做，但删除所有"为 AI 设计"的叙事

这个实验只需手写 CUDA + 现成 profiler + 一个 LLM API，6 周 3 人完全可以完成，成本不到全项目的 5%，却决定了剩余 95% 该不该花。

---

## 第二部分：语言设计

### 2.1 设计原则（用于裁决所有 API 争议）

1. **显式优先**：任何影响性能的决策，用户必须能显式指定；未指定时编译器用**固定、可预测、有文档**的默认值，不做搜索
2. **一个概念一个原语**：不允许 `ts.dot` / `ts.dot_tensorcore` / `ts.dot_simd` 三个并存。原方案的混用在此废止
3. **每个原语都有诊断挂点**：原语的 lowering 结果必须能在诊断 JSON 中以源码行号定位
4. **类型系统承担安全，不承担性能**：MemoryScope as Type 只保证不越界、不跨作用域误用；性能好坏交给用户和诊断

### 2.2 类型系统

```python
# 基础类型
dtype    ::= f16 | bf16 | f32 | f8e4m3 | i8 | i32
scope    ::= Global | Shared | Register | Distributed   # Distributed 见 §2.6
Tensor[dtype, shape, scope, layout=RowMajor]
Pointer[dtype, scope]

# 编译期常量
comptime[int]

# 作用域合法转移矩阵（编译期检查）
#            → Global  Shared  Register
# Global       -       load    load
# Shared       store   copy    load
# Register     store   store   move
# 其余组合：编译错误 E0301 "Invalid scope transfer"
```

**Pipeline 状态类型**（原方案用 dict 又用点号访问，此处正式定义）：

```python
@ts.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]
    m:     Tensor[f32, (BR,), Register]
    l:     Tensor[f32, (BR,), Register]
```

`@ts.state` 是一个纯数据结构体，所有字段必须是 Register scope（Pipeline 迭代间携带的状态不能藏在 Shared 里，否则同步语义不清）。

### 2.3 核心原语（统一版）

#### 内存操作

```python
ts.load(src, dst, mode=Sync|Async, group=None)   # 跨 scope 拷贝
ts.store(src, dst)
ts.commit_group() -> GroupHandle
ts.wait_group(handle | pending_count: int)
ts.barrier(scope=Block|WarpGroup)
ts.atomic_add(src: Register, dst: Global)
```

#### 计算原语——只有一个 dot

```python
ts.dot(
    A: Tensor[..., (M, K), Shared|Register],
    B: Tensor[..., (K, N), Shared|Register],
    C: Tensor[f32, (M, N), Register],
    *,
    mma: MMAShape | Auto = Auto,      # 显式指定或用硬件默认
    pad: PadPolicy = Error            # 见 2.4
) -> Tensor[f32, (M, N), Register]
```

- `mma=Auto` 时，编译器选择 HAL 报告的**第一个**可用形状（NVIDIA H200: m16n8k16；Ascend 910B: m16n16k16），这个选择是确定性的、写进文档的，并出现在诊断 JSON 里
- 用户/AI 可用 `mma=ts.MMA(16, 8, 32)` 覆盖，如果硬件不支持则编译错误 E0402，错误信息列出该硬件支持的所有形状
- 回退到 SIMD 路径**不会自动发生**。如果 tile 形状不能用 Tensor Core，报错 E0403 并建议 pad 策略。原则：编译器不替你降级性能

#### Warp 特化（一等公民）

```python
with ts.warp_group(role="producer", warps=2) as pg:
    # 只允许 load/commit/barrier，出现 dot 则编译错误 E0501
    ...
with ts.warp_group(role="consumer", warps=6) as cg:
    ...
ts.warp_group_sync(pg, cg, barrier_id=0)
```

在 Ascend 上，`warp_group` lower 为 AI Core 内 Vector/Cube 单元的角色分工（这在 Ascend 上是自然的——它本来就是异构单元），HAL 报告 `warp_group.max_roles`。

#### 归约

```python
ts.reduce(x, axis, op=Sum|Max, scope=Auto|Warp|Block)
```

`scope=Auto` 在 NVIDIA 上选 Warp（`__shfl_sync`），在 Ascend 上选 Block（无 warp 概念）。不提供 `warp_reduce_sum` 这种硬件专用名字——用户写 `if warp_size > 0` 分支的代码在原方案里出现过，这违背了 HAL 的意义，废止。

### 2.4 形状不匹配策略（对 Review 第二段疑问 1 的回应）

`ts.dot` 的 M/N/K 不能被 MMA 形状整除时，四种策略，用户必须显式选择：

| PadPolicy | 行为 | 性能影响 | 诊断中的体现 |
|-----------|------|----------|--------------|
| `Error`（默认） | 编译失败 | — | E0403，列出最近的合法形状 |
| `PadZero` | 编译器在 Shared 分配时补零到对齐 | 浪费 $\frac{\lceil M/m \rceil \cdot m}{M}$ 倍计算 | `dot.pad_waste_ratio` 字段 |
| `Mask` | 生成尾部 mask，最后一个 tile 用 predicated MMA | 每个 warp 多 1-2 条指令 | `dot.tail_masked: true` |
| `Split` | 对齐部分走 MMA，尾部走 FMA | 尾部性能骤降 | `dot.tail_fma_ratio` |

跨硬件一致性：同一份源码在 NVIDIA（k=16）和 Ascend（k=16）上 K 维对齐要求一致，N 维不一致（8 vs 16）。编译器在 `--target=all` 模式下会报告"N=24 在 NVIDIA 对齐、在 Ascend 需 pad 到 32"这类跨平台差异，作为诊断 JSON 的 `portability` 段。

### 2.5 Pipeline 抽象

```python
pipe = ts.Pipeline(stages=3, buffers={"K": K_shared, "V": V_shared})

@pipe.produce
def fetch(j: int, buf: ts.BufferSlot):
    ts.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)
    ts.load(V[j*BC:(j+1)*BC, :], buf.V, mode=Async)

@pipe.consume
def compute(j: int, buf: ts.BufferSlot, st: AttnState) -> AttnState:
    ...
    return st

state = pipe.run(range(num_blocks), init=AttnState(...))
```

**1.0 版本的 Pipeline 是"半自动"**：用户指定 stages 和 produce/consume 分工，编译器负责（a）多缓冲区分配（b）prologue/epilogue 展开（c）commit/wait 插入。用户**不能**手动写 `wait_group`——因为一旦允许手动，编译器就无法保证展开正确性，也无法形式化验证。

**诊断映射**：展开后每条 `cp.async` / `wait_group` 在诊断 JSON 中都带 `origin: {"pipeline": "pipe", "stage": "fetch", "iteration_class": "prologue|steady|epilogue", "src_line": 87}`。这回应了 Review 第三段疑问 2——展开不破坏可追溯性，因为展开时就把来源标签带着走。

**形式化验证**：Pipeline 展开 pass 输出缓冲区生命周期约束，由 Z3 验证无 use-before-ready / overwrite-before-consumed。1.0 必须包含，不作为可选项。

### 2.6 Persistent Kernel 与跨 Kernel 融合

```python
@ts.persistent_kernel(scheduler=ts.TileScheduler.RoundRobin)
def gemm_persistent(...):
    for tile in ts.tile_iter():     # 每个 CTA 循环领取 tile
        ...

@ts.fused_kernel
def gemm_gelu(...):
    C = gemm_body(...)
    return ts.elementwise(C, gelu)  # 融合到 epilogue，不落 Global
```

这两项 1.0 只在 NVIDIA 上提供；Ascend 的 persistent 语义需要 CANN 侧的 task scheduler 配合，标注为 1.1。

---

## 第三部分：结构化诊断系统

这是项目的核心交付物，与编译器同等重要。原方案只有一句话提到"JSON 报告"，这里给出完整规范。

### 3.1 诊断 JSON Schema（v1，冻结）

```json
{
  "schema_version": "1.0",
  "target": "nvidia_h200",
  "kernel": "flash_attn_fwd",
  "compile_time_ms": 2140,
  "deterministic_hash": "sha256:...",

  "resources": {
    "registers_per_thread": 168,
    "register_spill_bytes": 0,
    "shared_memory_bytes": 98304,
    "occupancy": {"active_warps_per_sm": 8, "limiter": "shared_memory"}
  },

  "memory": [
    {
      "tensor": "K_shared", "src_line": 42,
      "layout": "swizzled(xor=0b11100)",
      "bank_conflict": {"read_way": 1, "write_way": 2,
                        "worst_access_line": 88}
    }
  ],

  "async": [
    {
      "pipeline": "pipe", "stages": 3,
      "estimated_gap_cycles": 340,
      "wait_coverage": 0.91,
      "note": "wait_group at line 71 blocks 9% of steady-state iterations"
    }
  ],

  "compute": [
    {
      "op": "dot", "src_line": 95,
      "mma_shape": "m16n8k16", "tensor_core": true,
      "pad_policy": "Mask", "tail_masked": true,
      "instruction_count": 128
    }
  ],

  "portability": [
    {"issue": "N=72 requires pad to 80 on ascend_910b (m16n16k16)",
     "src_line": 95, "severity": "warning"}
  ],

  "suggestions": [
    {"code": "S0201", "src_line": 88,
     "msg": "2-way write bank conflict; try xor mask 0b10100 or layout=Padded(8)"}
  ]
}
```

### 3.2 诊断的数据来源与精度承诺

| 字段 | 来源 | 精度 |
|------|------|------|
| registers / spill / shared | `ptxas -v` / CANN 编译日志 | 精确 |
| bank_conflict | 编译器静态分析（已知 layout + 访问模式） | 静态可判定的场景精确；动态索引标注 `unknown` |
| occupancy | 由 registers/shared 查表计算 | 精确（理论值） |
| async gap | 静态估算（load 字节数 / HAL 带宽 - compute 指令数 × IPC） | **粗估**，明确标注 `estimated` |
| 运行时 stall reason | 可选 `--profile` 模式调 Nsight Compute / Ascend Insight | 精确但慢 |

**关键诚实性声明**：静态诊断不能替代 profiling。TileScript 的定位是"静态诊断让 AI 快速排除明显问题（bank conflict、spill、占用率），每 N 轮再用 `--profile` 拿真实数据"。原方案 5.4.1 那个 roofline 性能模型在此降级为诊断中的 `estimated_bound: compute|memory` 一个字段，不再承担"预测性能"的角色。

### 3.3 AI 优化循环的参考实现

```python
# tilescript/agent_loop.py —— 随 1.0 发布，是 H1 验证的正式版本
def optimize(kernel_src, target, budget_rounds=20, llm=...):
    best = compile_and_bench(kernel_src, target)
    for r in range(budget_rounds):
        diag = ts.diagnose(kernel_src, target, profile=(r % 5 == 0))
        proposal = llm.propose(kernel_src, diag, history)
        result = compile_and_bench(proposal, target)
        if not result.ok:              # 编译错误也是结构化的，回给 LLM
            history.append((proposal, result.errors)); continue
        if result.perf > best.perf: best, kernel_src = result, proposal
        history.append((proposal, diag, result.perf))
    return best
```

这段代码不是 AutoTuner。原方案两个签名不一致的 `AutoTuner` 全部删除。1.0 **不发布**参数网格搜索工具——如果用户想要，Triton 那套已经很好；我们的差异化是这个 agent loop，而且它的效果在阶段0就要有数据。

---

## 第四部分：硬件抽象层与后端

### 4.1 HAL 能力描述（静态 YAML，编译期读取）

```yaml
# hal/nvidia_h200.yaml
name: nvidia_h200
warp_size: 32
smem_bytes: 232448
smem_banks: 32
registers_per_sm: 65536
async_copy: [cp_async, tma]
mma_shapes: [[16,8,16], [16,8,32]]      # 顺序即 Auto 的默认选择顺序
warp_group: {supported: true, max_roles: 2}
reduce_scopes: [warp, block]
persistent_kernel: true

# hal/ascend_910b.yaml
name: ascend_910b
warp_size: 0                              # 无 warp 概念
smem_bytes: 524288                        # L1 Buffer
ub_bytes: 262144                          # Unified Buffer，独立层级
smem_banks: 16
async_copy: [dma]
mma_shapes: [[16,16,16]]
warp_group: {supported: true, max_roles: 2, roles: [vector, cube]}
reduce_scopes: [block]
persistent_kernel: false                  # 1.1
```

### 4.2 内存层级映射（含 UB）

Ascend 的 UB 不再由编译器"自动选择"（原方案的 `if data_size < 64KB` 启发式违背了设计原则 1）。改为：

- `Shared` scope 在 Ascend 上默认映射到 L1 Buffer
- 用户可显式写 `ts.alloc_shared(..., hint=ts.Placement.Fast)`，在 Ascend 映射到 UB，在 NVIDIA 无操作
- 诊断 JSON 的 `portability` 段会提示"此 tensor 大小 48KB 且流式访问，在 Ascend 上建议 `Placement.Fast`"——**建议由诊断给出，决策由用户/AI 做**

### 4.3 后端实现策略

| | NVIDIA | Ascend |
|---|---|---|
| Lowering 路径 | TSR → MLIR NVVM dialect → PTX | TSR → AscendC C++ 源码 → CANN 编译 |
| 复用基础设施 | MLIR GPU/NVVM 现成 | 自建 C++ emitter（约 4k 行） |
| 异步拷贝 | cp.async / TMA | DataCopy + SetFlag/WaitFlag，event 由 Pipeline pass 分配 |
| dot | wmma / mma.sync intrinsic | AscendC Matmul API |
| barrier | bar.sync | PipeBarrier(PIPE_ALL) |
| 诊断来源 | ptxas -v + 静态分析 | CANN 编译日志 + 静态分析（bank 模型需 M1 验证） |

Ascend 后端"生成 C++ 而非 MLIR"意味着**无法复用 MLIR 的优化 pass**，这是原方案低估工作量的根源。新方案的处理：Ascend 后端只实现 lowering 和诊断，**不做任何 Ascend 专属优化 pass**，性能好坏完全交给 CANN 编译器 + 用户显式控制。这就是为什么 Ascend 1.0 定为"实验级"。

---

## 第五部分：团队与路线图（重新分配）

### 5.1 团队（10 人）

| 角色 | 人数 | 阶段0 | M1-M10 |
|------|------|-------|--------|
| 语言/类型系统/前端 | 2 | 参与阶段0实验设计 | 前端 + Pipeline pass + Z3 验证 |
| NVIDIA 后端 | 2 | 手写 CUDA 原型（E1/E2） | MLIR lowering + 诊断静态分析 |
| 诊断系统 + AI loop | 2 | 诊断插件（C1）+ agent loop | 诊断 JSON、agent loop、对照实验持续运行 |
| Ascend 后端 | 2 | H2 Go/No-Go 验证 | AscendC emitter + CANN 对接 |
| PyTorch 集成 + 测试/CI | 1 | CI 搭建 | Dispatcher/Autograd/构建 |
| 项目负责人 / 兼 kernel 工程师 | 1 | 主导阶段0判定 | 5 个标准 kernel 编写、benchmark |

对比原方案：NVIDIA 组从 3 人减到 2 人（MLIR 复用度高），**新增 2 人专职诊断 + AI loop**（原方案 0 人——项目核心却没人负责，这是最大的资源配置错误）。

### 5.2 里程碑

```
阶段0 (W1-W6)   假设验证：H1/H2/H3 三个 Go/No-Go
                 └─ 任一 No-Go → 按 §1.3 处置，不进入 M1

M1  (M1)         前端：Parser(Python AST) / 类型检查 / TSR / 作用域检查
                 验收：5 个 kernel 源码全部通过类型检查；故意注入的 20 个作用域错误全部被捕获

M2-M3 (M2-M3)    NVIDIA lowering + 诊断 v0
                 验收：vector_add/matmul 正确；诊断 JSON 的 resources/memory 段与 ptxas/Nsight 一致
                       matmul 4096³ ≥ cuBLAS 50%（无 pipeline）

M4-M5 (M4-M5)    Pipeline pass + Z3 验证 + Warp 特化 + 诊断 v1（async/compute 段）
                 验收：matmul ≥ cuBLAS 65%；FlashAttention fwd 正确；
                       agent loop 在 matmul 上 20 轮内从 65% 提升到 ≥ 75%（H1 的中期复验）

M6  (M6)         PyTorch 集成（CUDA）
                 验收：FA fwd+bwd autograd 梯度检查通过；tilescript_ops 可 pip 安装

M7-M8 (M7-M8)    Ascend 实验后端（前提 H2=Go）
                 验收：vector_add/matmul/layernorm 双平台数值一致（rtol 1e-5 / 1e-2 / 1e-4）；
                       Ascend matmul ≥ CANN 50%（注意：比原方案 65% 下调，因为不做专属优化 pass）
                       诊断 portability 段能正确报告 N 维对齐差异

M9-M10           打磨 + 发布
                 验收：见 §5.3
```

### 5.3 1.0 验收标准（全文唯一一份数字，其他章节不得另写）

| 指标 | NVIDIA H200 | Ascend 910B（实验级） |
|------|-------------|----------------------|
| MatMul 4096³ FP16，裸性能 | ≥ 65% cuBLAS（≥ 580 TFLOPS） | ≥ 50% CANN |
| MatMul，agent loop 20 轮后 | ≥ 80% cuBLAS | 不作要求 |
| FlashAttention fwd seq=2048 | ≥ 65% FA2 官方 | 不支持 |
| FlashAttention，agent loop 后 | ≥ 78% FA2 官方 | — |
| LayerNorm 带宽 | ≥ 70% 理论 HBM | ≥ 50% |
| 编译时间（FA fwd，≈150 行） | ≤ 5 秒 | ≤ 20 秒（含 CANN） |
| 确定性 | 100 次编译 hash 一致 | 同 |
| 诊断准确率 | bank conflict 静态判定与 Nsight 一致率 ≥ 90% | — |

原方案"< 3 秒"改为"≤ 5 秒"：加了 Z3 验证后 3 秒不现实，宁可承诺低一点。

---

## 第六部分：风险登记表（修订）

| 风险 | 概率 | 影响 | 缓解 | 触发处置 |
|------|------|------|------|----------|
| **H1 不成立**（AI+诊断不比启发式好） | 中 | 致命 | 阶段0 6 周实验 | 终止项目或转为 Triton 插件（3 人） |
| **H3 不成立**（插件就够了） | 中 | 致命 | 阶段0 E1 vs C1 对照 | 转为 Triton 插件 |
| **H2 不成立**（Ascend 走不通） | 高 | 高 | 阶段0 4 周验证 | 1.0 去掉 Ascend，**叙事改为"AI 辅助优化"单支柱**，不转 AMD（AMD 不是差异化，1.1 再做） |
| Ascend 文档不足 | 高 | 中 | 华为技术支持、招 CANN 经验者 | 同上 |
| Pipeline pass 正确性 | 中 | 高 | Z3 形式化验证 | 降级为"用户显式写 wait，编译器只检查"（性能不变，易用性降） |
| 静态 bank conflict 分析覆盖不足 | 中 | 中 | 动态索引标 unknown，引导用 --profile | 诊断准确率指标下调至 80% |
| 团队规模 | 中 | 中 | Ascend 已降为实验级，不做自动优化 | 砍 Softmax/Transpose 两个 kernel |

关于原方案"Ascend 失败转 AMD"：新方案不这样做。理由是 Review 第五段第 4 点——AMD 不构成差异化，转 AMD 只是为了让"跨硬件"叙事看起来还在，但那是**为叙事服务而不是为价值服务**。如果 H2 失败，诚实地做 NVIDIA 单后端，把"AI 辅助优化"做透，比硬凑一个第二后端更有价值。

---

## 第七部分：完整案例——FlashAttention Forward（统一后的 API）

```python
module flash_attention:

@ts.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]
    m:     Tensor[f32, (BR,), Register]
    l:     Tensor[f32, (BR,), Register]

@ts.kernel
def flash_attn_fwd(
    Q_ptr: Pointer[f16, Global], K_ptr: Pointer[f16, Global],
    V_ptr: Pointer[f16, Global], O_ptr: Pointer[f16, Global],
    L_ptr: Pointer[f32, Global],
    seq_len: int, scale: f32,
    D: comptime[int] = 64, BR: comptime[int] = 64,
    BC: comptime[int] = 64, STAGES: comptime[int] = 2
):
    bm = ts.block_idx(0)
    Q = ts.make_tensor(Q_ptr, (seq_len, D)); K = ts.make_tensor(K_ptr, (seq_len, D))
    V = ts.make_tensor(V_ptr, (seq_len, D)); O = ts.make_tensor(O_ptr, (seq_len, D))
    L = ts.make_tensor(L_ptr, (seq_len,))

    Q_s = ts.alloc_shared((BR, D), f16, layout=ts.Layout.swizzled(xor=0b11100))
    K_s = ts.alloc_shared((BC, D), f16, layout=ts.Layout.swizzled(xor=0b11100))
    V_s = ts.alloc_shared((BC, D), f16, layout=ts.Layout.swizzled(xor=0b11100))

    ts.load(Q[bm*BR:(bm+1)*BR, :], Q_s, mode=Sync)
    ts.barrier()

    pipe = ts.Pipeline(stages=STAGES, buffers={"K": K_s, "V": V_s})

    @pipe.produce
    def fetch(j: int, buf):
        ts.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)
        ts.load(V[j*BC:(j+1)*BC, :], buf.V, mode=Async)

    @pipe.consume
    def attend(j: int, buf, st: AttnState) -> AttnState:
        S = ts.dot(Q_s, ts.transpose(buf.K), ts.zeros((BR, BC), f32, Register),
                   mma=ts.MMA(16, 8, 16), pad=ts.PadPolicy.Error)
        S = S * scale
        m_ij  = ts.reduce(S, axis=1, op=Max)
        m_new = ts.maximum(st.m, m_ij)
        alpha = ts.exp(st.m - m_new)
        P     = ts.exp(S - m_new[:, None])
        l_new = alpha * st.l + ts.reduce(P, axis=1, op=Sum)
        O_new = alpha[:, None] * st.O_acc + ts.dot(ts.cast(P, f16), buf.V,
                                                   ts.zeros((BR, D), f32, Register))
        return AttnState(O_acc=O_new, m=m_new, l=l_new)

    st = pipe.run(range(ts.cdiv(seq_len, BC)),
                  init=AttnState(O_acc=ts.zeros((BR, D), f32, Register),
                                 m=ts.full((BR,), -inf, f32, Register),
                                 l=ts.zeros((BR,), f32, Register)))

    ts.store(ts.cast(st.O_acc / st.l[:, None], f16), O[bm*BR:(bm+1)*BR, :])
    ts.store(ts.log(st.l) + st.m, L[bm*BR:(bm+1)*BR])
```

相对原方案的修正：
- 归一化 $O / l$ 移到循环外一次完成（原方案每轮都除 `l_new`，数学上等价但多做了 $N/BC$ 次除法，也是 FA2 相对 FA1 的改进点之一）
- `P` 在喂给 `dot` 前显式 cast 到 f16——原方案传 f32 的 P 给 Tensor Core，在 NVIDIA 上不合法
- 去掉所有 `if warp_size > 0` 分支

---

## 第八部分：与现有方案对比（精简、可验证）

| 维度 | Triton | TileLang | CuTe | TileScript 1.0 |
|------|--------|----------|------|----------------|
| 优化决策者 | 编译器 | 编译器 | 用户 | 用户 + AI |
| 结构化诊断（源码级） | 无 | 无 | 无 | 有（JSON Schema v1） |
| MMA 形状可指定 | 否 | 否 | 是 | 是 |
| Warp 特化可表达 | 否 | 自动注入 | 是 | 是 |
| 确定性编译 | 否 | 否 | 是 | 是 |
| 编译期作用域检查 | 否 | 否 | 部分（模板） | 是 |
| 裸性能（无人工/AI调优） | 高 | 高 | 取决于用户 | **中**（65%） |
| AI 辅助后性能 | 参数搜索上限 | 参数搜索上限 | 无诊断，难 | **目标 80%，阶段0 出数据** |
| Ascend | 否 | 适配器 | 否 | 实验级 |
| 成熟度 | 生产 | 接近生产 | 生产 | MVP |

不再列"性能上限 85%/90%/95%"这类无来源的数字。

---

## 第九部分：明确剥离的内容

以下内容从技术方案中移除，另立文档：
- **商业模式与收入预测** → 《TileScript 商业计划书》，在阶段0 Go 之后再写
- **专利申请** → 移除。三项声明的在先技术过多（CUTLASS 的 tiled copy 抽象、Triton 的 pipeline pass、TVM 的 layout 推导），申请成功概率低且消耗团队精力；防御性目的用 Apache 2.0 + 公开发表即可达成
- **开源社区运营节奏** → 简化为一句：阶段0 Go 后开源诊断插件（因为它对 Triton 用户也有价值，能先建立社区），M6 开源编译器主体

---

## 结语：这版方案与上一版的本质区别

上一版的逻辑是："我们有一个好想法 → 这是 10 个月的计划 → 这是风险和应对"。

这一版的逻辑是："我们有一个好想法 → **这是花 5% 成本验证它是否成立的方法** → 成立了再看 10 个月计划"。

三个假设、一份数字、一个 dot、一套诊断 Schema。如果阶段0 结束时 H1/H3 不成立，这份文档会被诚实地降级成一份 Triton 插件的设计说明——那也是一个有价值的产出，只是没那么激动人心。**能被证伪的方案，才是值得投入的方案。**
