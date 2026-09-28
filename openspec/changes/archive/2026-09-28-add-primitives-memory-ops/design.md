# Design

## 设计范围

| 受影响 capability | 目标变化 | delta specs | 设计章节 |
|---|---|---|---|
| primitives/memory-ops | 新增核心存取原语行为契约（纯规格，无代码交付） | specs/primitives/memory-ops/spec.md | 第 1 节 |
| language/type-system | MODIFIED「类型不匹配拒绝」：等价豁免边界扩展至全部 `tis.*` 原语实参 | specs/language/type-system/spec.md | 第 1 节 |

## 1. primitives/memory-ops

### 目标与规范依据

目标 Requirements：`specs/primitives/memory-ops/spec.md` 的全部 ADDED Requirements——原语集合与调用结构、转移格承载映射、load/store 数据维度契约、加载模式与完成语义、分配与视图构造、dtype 显式转换、屏障语义、报告契约与段位。

规范依据：`veps/design.md` §2.1 原则 1（显式优先：数据移动与转换必须显式）与原则 5（HAL 唯一硬件知识来源）、§2.3 内存操作清单（六原语签名）、§2.5（Pipeline 半自动：用户不能手动写 `wait_group`）、§3.1（`Layout.swizzled`/`Padded` 的诊断侧提及）、§7（十类原语用例）。

### 当前实现

仓库尚无编译器代码。原语事实只存在于 `veps/design.md`：§2.3 内存操作清单六条单行签名（`load`/`store`/`commit_group`/`wait_group`/`barrier`/`atomic_add`），参数级契约（返回类型、dtype/shape 约束、mode 语义、layout 值域）全部未定义；§7 大量使用的 `make_tensor`/`alloc_shared`/`zeros`/`full`/`cast` 根本不在 §2.3 清单内（用例级事实）。既有错误码事实：`E0402`/`E0403`（MMA/pad）、`E0501`（warp）、`E0101`–`E0107` 与 `E0301`–`E0304`（已归档 specs）。`language/type-system` 两处让渡（移动原语实参其余维度、`tis.*` 调用结果类型）指向 `primitives/*`。

### GAP 分析

| 规范目标 | 当前事实 | 差距 |
|---|---|---|
| 原语集合与参数结构 | §2.3 仅六条单行签名；§7 五原语不在清单 | 参数集、默认值、封闭性（未知实参拒绝）均未定义 |
| 转移格承载 | type-system 已定义矩阵与格名（load/store/copy/move） | 各格由哪个原语承载未定义；copy/move 格悬空 |
| 数据维度契约 | type-system 豁免移动原语实参的 E0303，让渡本域 | dtype/shape 一致性规则与错误码缺失 |
| mode 与完成语义 | §2.3 签名含 `mode=Sync\|Async, group=None`；§2.5 禁手动 wait_group | Async 完成语语与合法语境未定义；group 与 commit/wait 的用户可见性存在设计张力 |
| 分配与构造 | §7 用例（make_tensor/alloc_shared/zeros/full）无签名定义 | 返回类型、scope/dtype/layout 值域、value 约束缺失 |
| cast | §7 两处用例；type-system 非目标声明归本域 | 返回类型规则与显式性声明缺失 |
| barrier | §2.3 签名 `scope=Block\|WarpGroup`；§7 无参调用 | 值域默认值与汇合/可见性语义未定义 |
| E04xx 段位 | 仅 E0402/E0403（MMA）两事实 | 存取原语违规无码；段位语义与 tiebreak 未定义 |
| E0303 豁免边界 | type-system 豁免仅限数据移动原语（初始集合 load/store） | 构造/转换原语实参存在 E0303/E0406 双解，边界未定义 |

### 修改方案

本 change 为纯规格 change，无代码交付。关键裁决：

- **原语集合裁决（八原语）**：`load`/`store`/`barrier`（§2.3 清单内、§7 使用）+ `make_tensor`/`alloc_shared`/`zeros`/`full`/`cast`（§7 用例升格为权威签名）。`commit_group`/`wait_group`/`atomic_add` 延期：`atomic_add` M1 零使用；`commit_group`/`wait_group` 的 `GroupHandle` 在 type-system 封闭类型世界（三类标量 + Tensor + Pointer + 状态类）之外，引入需另行裁决其类型层地位，且 §2.5 明确 Pipeline 语境用户不能手动写 `wait_group`——独立 async change 承载，本 change 内传 `group=` 按未知参数 `E0406` 拒绝（保守封闭）。
- **Async 仅 Pipeline 语境（E0407）**：`mode=Async` 只在 `produce`/`consume` 嵌套函数体内合法。依据：§7 事实（Async 仅出现于 produce）；完成保证需要 commit/wait 机制，而该机制在 1.0 由 Pipeline 编译器插入（§2.5），用户无法在非 Pipeline 语境提供完成保证——允许将产生无法验证的悬空读。完成语义让渡执行结构 capability。
- **格子承载映射**：`load` 承载全部三个 load 格、`store` 承载全部三个 store 格（一原语多格，按实参 scope 组合分发）；`copy`/`move` 格显式声明无承载原语（veps 无事实，不虚构）。`E0404`（原语-格类别不匹配）与 `E0301`（矩阵非法格）的边界：scope 组合先过 type-system 矩阵（非法格 → `E0301`），合法格再过本映射（类别错 → `E0404`）。`E0404` 的裁决对象限定为数据移动原语（load/store）的 scope 组合（单实参原语不构成转移组合）；同 scope 数据路径由普通赋值承载（type-system 类型等价裁决），copy/move 格的专用显式原语延期（设计审查 round 1 minor 3 修正表述）。
- **数据维度契约（E0405，设计审查 round 1 F2）**：dtype 相同 + shape 逐维**长度相容**——本 capability 专用判定，不沿用 type-system 维度等价（该规则无"运行期派生维 vs 编译期常量维"等价支，会使 §7 全部 tile 切片移动必命中 E0405）。逐维四支：常量值等/同一运行期符号/派生表达式结构等价/派生长度可静态折叠为常量且与对侧值相等（如 `bm*BR:(bm+1)*BR` 折叠为 `BR`）。长度相容仅用于移动原语实参，不产生类型层等价结论。`load`/`store` 实参必须 Tensor：`Pointer` 直接传给 `load` 拒绝（`E0406`）——§7 惯例是先 `make_tensor` 建视图再切片移动，Pointer 快捷路径无事实，不引入。Pipeline buffer 属性（`buf.K`）的结果类型显式让渡执行结构 capability，本 change Scenario 仅以其已是 Shared Tensor 为前提（设计审查 round 1 minor 5）。
- **无值原语（E0407）**：`load`/`store`/`barrier` 合法形态仅为表达式语句；type-system 的类型世界无 void 种类，值使用按语境违规拒绝（与 Async 语境同码，均为"合法语境"类）。
- **make_tensor 限 Global Pointer**：语义是设备指针的零拷贝 Global 视图（§7 全部五例均 Global）；`Pointer[d, Shared]` 无语义事实，拒绝（`E0406`）。动态 shape 维（`seq_len`）成为返回 Tensor 的运行期维度——承接 type-system R2 动态维度规则。
- **zeros/full scope 限 Register|Shared 且默认 Register**：Global 张量必须经 `make_tensor` 从指针建立（显式数据来源），分配构造原语不提供"凭空 Global"；默认 Register 对齐"计算默认域"与 §7 用例。默认值固定写进文档（对齐原则 1 的默认值确定性要求）。`full` 的 `value` 形态覆盖三类标量种类、数学具名常量及其一元负表达式（如 `-inf`，设计审查 round 1 minor 2），数值转换延期数值语义。
- **cast 全六 dtype 开放**：结构上任意两 dtype 间显式 cast 合法（显式优先——用户显式要求即执行，编译器不替用户判断"值不值得"）；数值效果（舍入/饱和/精度损失）延期数值语义。标量 cast 无事实（§7 仅 Tensor），拒绝（`E0406`）。算术结果表达式作为 `x` 时其结果类型依赖数值语义 capability——该 capability 定义前仅形态级核对（设计审查 round 1 minor 6，依赖顺序记入 tasks）。
- **位置实参传递（设计审查 round 1 F1）**：可省略参数可按声明顺序以位置实参传递（§7 zeros/full 事实——`tis.zeros(shape, dtype, Register)`、`tis.full(shape, value, dtype, Register)`）；拒绝条件为位置实参数量超出参数集总长度（必选加可省略），非超出必选数（原表述与 §7 用例及自身 Scenario 互斥）。
- **E0303 豁免扩展（设计审查 round 1 F4，MODIFIED）**：本 change 含 `language/type-system`「类型不匹配拒绝」的 MODIFIED delta——等价豁免从"显式数据移动原语（初始集合 load/store）"扩展为"`tis.*` 原语调用表达式的实参整体"。理由：原语无源码层形参注解，实参约束是参数契约（形态+值域）而非类型化绑定，等价规则无从落地；type-system 埋有"权威集合由 primitives/* 定义"钩子，本 change 行使该扩展并显式化。BREAKING 标记：`E0303` 适用集收敛（原语实参不再可能命中 E0303），无存量实现与受众，迁移路径为无。
- **swizzled 非类型组件（设计审查 round 1 F3）**：`swizzled` 是分配的物理布局属性，记录于资源诊断 `memory` 段（建议域，S0201 类——与 spec 术语统一），不进 Tensor 类型 layout 组件——type-system"layout 唯一合法值 RowMajor"的封闭声明无需修改，swizzled 分配的结果 Tensor 类型组件为 `RowMajor`，不影响类型等价判定。xor 值约束：`comptime[int]` 且非负（§7 例 `0b11100` = 28，非 2 的幂——不引入 2 的幂约束）；掩码与行宽的可满足性不做编译期拒绝。`Padded` 布局仅在诊断建议文本出现，值域延期。
- **barrier 语义**：值域 {Block, WarpGroup}（§2.3 事实），默认 Block（§7 无参调用）；Block 模式定义汇合与跨线程可见性（Sync load 的可见性补充）；WarpGroup 模式的上下文约束（E0501 域）与汇合/可见性语义均让渡执行结构 capability（设计审查 round 1 minor 1 补汇合语义归属）。
- **E04xx 段位语义精化**：段位解释从"计算原语"精化为"原语参数契约段"（计算原语是其子域）——与既有 `E0402`/`E0403` 不冲突且同化段位分配原则（段位=检查阶段：语法→类型→原语契约→执行结构）。新码按违规类别分配：`E0404`（格类别）/`E0405`（维度）/`E0406`（参数值域形态）/`E0407`（语境）；`E0408`–`E0499` 保留。
- **段内 tiebreak**：`E0404`→`E0405`→`E0406`→`E0407`（先裁"哪个操作/格子"，再裁"数据对不对"，再裁"参数值合法否"，最后裁"位置对否"）；跨段 tiebreak 延伸 type-system 规则为 E01xx > E03xx > E04xx。
- **充分性标准**：覆盖 §7 全部十类原语用法（load×2 模式、store×2、barrier、make_tensor×5、alloc_shared×3（swizzled）、zeros×4、full（-inf）、cast×2（Tensor/派生表达式）、Layout.swizzled 参数——zeros 计数经代码审查 round 1 修正），核对任务见 tasks。
- **确定性影响**：本 change 无实现，不改变任何 `deterministic_hash`；"重复编译 E04xx 清单一致"为将来实现固化确定性要求。全部检查为编译期静态契约（参数形态、值域、格子、语境），不涉及运行时反馈（对齐编译器定位：本批原语的拒绝均为可静态化反馈）。

质量属性影响：无新增黑盒质量目标（可验证性由各 Requirement 的 Scenario 承载）。

## 长期基线刷新计划

- stable specs：归档时新增 `openspec/specs/primitives/memory-ops/spec.md`；按本 change 的 MODIFIED delta 更新 `openspec/specs/language/type-system/spec.md`——「类型不匹配拒绝」Requirement 以豁免边界扩展后的完整重述替换既有块（Scenario 标题保持 canonical 不变）。
- designs：无（M1 原语实现设计由实现 change 承载）。
- overview：在「稳定基线」节登记 `primitives/memory-ops` 索引。
