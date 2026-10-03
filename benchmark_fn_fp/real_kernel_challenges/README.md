# 专门构造的真实 kernel 验证挑战

2026-10-03。该组是**针对验证机制专门准备的研究挑战集**，不是从生产
kernel 随机抽取的代表性样本。公开 kernel 来自真实缺陷机制或历史修复，
适配、人工修改及测试选择均记录；本组结果独立统计，不与旧题库合并为总准确率。

## 实验后的保留分组

当前优先比较**带工具验证流程与无工具单次调用**。成功案例按实际观察到的判断
差距及合法验证证据保留；暂不把纯工具因果归因或 debate 超过 solo 作为入选门槛。
按此口径，**case_114 继续归入成功案例**。原始结果与证据审计仍完整保留。

| 分组 | Case | 后续用途 |
|---|---|---|
| 成功案例：已观察到工具流程增益，单独保留 | **case_109、case_114** | 109 的 solo、114 的 solo/debate 正确拒绝，无工具错误接受；保留为后续扩展与复测的参考 |
| 未体现增益，保留待归普通集合 | case_106、107、108、110、111、112、113、115 | 保留正确/错误 kernel、来源和全部 traces，后续整理进普通集合 |

整理状态登记在 `case_map.json` 的 `curation_status`，并显示于案例和 GLM 索引。
当前仅分组保留，未正式迁移普通集合；编号、公开题目路径、固定标签和 traces
均保留。下方 106–111 六例与 112–115 四例分别完整统计，不能仅用筛出的 109/114 重算成功率。

## 固定标签与公开合同

一个 case 对应固定实现、固定环境及完整公开合同，因此只有一个 accept/reject
标签（现有程序用 `trust` 表示 accept）。合法输入是检验实现、构造反例的材料，
换输入不改变标签。正确／错误对必须是不同代码实现；不会把同一 kernel 的一个
通过 seed 与另一个失败 seed 当成一对正确／错误案例。

先确定合同、来源、候选实现及预期标签，再独立执行资格验证，最后才调用评测
模型。拒绝标签需要合法、可复现的合同违反；接受标签需要实现分析、独立参考
及边界测试支持。有限测试不是全输入域形式证明，测试覆盖也不定义为全部合法输入。
不根据模型是否答错修改本轮标签、阈值或筛掉案例。

## 首轮范围与存放位置

初始计划是两种机制各两个不同实现，`case_106`–`case_109`，共 12 个模型实验槽位。
独立补审发现两个原拟接受版本还有 wrapper 缺陷，因此保留它们并修正错误真值，
另增修复实现 `case_110`–`case_111`。最终为 **6 例 × 3 组 = 18 个模型槽位**。
这两个追加案例是补审后的修复对照，不能说成预先登记的六例随机样本。
每个新实现完成 GPU 资格检查后才进入付费评测；未覆盖的合法输入类导致的资格
遗漏及后续修正明确披露，原始模型输出不改写。

| 编号 | 机制 | 真实来源与适配 |
|---|---|---|
| case_106–107 | 状态滑动窗口原地更新，跨 GPU block 重叠读写 | PyTorch issue #164701 的机制；具体移植及修改在私有 provenance 中说明 |
| case_108–109 | RMSNorm backward 权重梯度归约精度 | Liger-Kernel PR #950；适配 T4 的 dtype 路径，不称 A100/BF16 原样复现 |
| case_110 | 状态更新的完整修复 | 独立输出、逻辑视图解析和独立 Triton 位级写回；合同与 106–107 相同 |
| case_111 | RMSNorm 的完整修复 | FP32 归约并解析四个输入的逻辑负号；合同与 108–109 相同 |

## 独立资格检查与真值修正

以下是作者侧 T4 验证，不是模型自行获得的成绩。证据保存在
`private_data/qualification/`；私有目录不会提供给模型。

| Case | 当前唯一标签 | 独立依据 |
|---|---|---|
| 106 | reject | 原地并行滑移，201 次调用中 119 次位级错误；row-markers 大形状 64/64 次失败 |
| 107 | reject | 独立输出消除了目标 race，但合法 requires_grad 叶子使 torch.copy_ 报错；补审还覆盖 inference/负视图状态 |
| 108 | reject | FP32 归约通过原 18 组数值检查，但合法 lazy-negative 输入被 raw pointer 错读；10 组视图测试均失败 |
| 109 | reject | 原 18 组有 9 组误差超界；明确梯度反例误差 0.0625，允许 0.002570625，约 24.31 倍 |
| 110 | accept（程序中为 trust） | 分离读写与逻辑符号位处理的源码分析；201 普通调用＋57 张量状态/连续调用全部通过 |
| 111 | accept（程序中为 trust） | FP32 误差界、逻辑视图解析；原 18 组＋10 组视图检查全部通过 |

**107、108 最初的 accept 是作者资格检查遗漏，不是两个同时有效的输入相关标签。**
原合同没有排除带梯度、inference 或逻辑负号标志的普通张量。发现合法反例后，
两例在原合同下统一纠正为 reject；没有把合同改窄，也没有改原实现。旧 r2/r3
资格记录保留为被补审推翻的历史判断，`validation_gpu.json` 保存当前标签与
`label_correction`，明确指向新证据 r4/r5。所有历史模型判断据当前正确真值评分。

例如 108 的合法负视图输入在 M=N=31 时，某个 dw 应为 -23.25，实际为 +23.25；
误差允许值仅 0.0002425。107 的 requires_grad 叶子反例尺寸为 [2,17,33]、K=3，
直接触发原地写回异常。这些额外缺陷来自本轮独立接口适配，不能说成上游原项目
已经被证实存在的同类问题。110/111 保留完整原合同，通过改代码修复。

- [统一公开案例](../triton_eval_cases/)：每例 `kernel.py` 包含完整候选、wrapper、
  公开参考及必要依赖；`problem.txt` 描述整个合法域和精度要求。
- [构造、资格验证和报告程序](../eval_scripts/real_kernel_challenges/)。
- `private_data/`：来源版本、适配与语义修改、预先声明的标签、资格验证记录及源码冻结。
  这些文件不会挂载到模型执行环境，也不会作为提示输入。
- [GLM traces](../traces_glm/)：`case_NN/{single_call,solo,debate}/rN/`。
  失败和重试均保留，不覆盖、不挑最好的一次。

编号与类型统一登记在 [case_map.json](../case_map.json) 和
[CASE_INDEX.md](../CASE_INDEX.md)。本 README 集中保存该组说明、结果及分析。
原 [PLAN.md](PLAN.md) 是 2026-09-27 的来源初筛计划；本轮范围和协议以此页为准。

## 三组协议

三组均使用 Fireworks `accounts/fireworks/models/glm-5p3`、reasoning effort `low`，
获得完整相同的 `kernel.py` 与 `problem.txt`。无工具组一次调用；solo 是一个
可调用工具的 agent；debate 使用现有 describer、skeptic、experimenter、judge 四角色。
两个工具组均运行于 T4、PyTorch 2.8.0、Triton 3.4.0、NumPy 1.26.4。

每题每组累计输出上限为 32,768 tokens；debate 的所有角色共享该上限。
记录实际输入、输出、工具执行与费用；同输出上限不代表同成本。
本轮修复了 `request_more_debate` 不重新打开未决 claim 的调度问题，并做了离线
回归；运行代码指纹保存在每条 trace 中，因此本轮不能与旧 runtime 直接混为
同一对照实验。三组公共源码无截断的一致性已有离线检查。

原始请求与响应、usage、最终判定、完整交互、probe 代码和 stdout/stderr 均保留。
模型置信度仅作为原始输出记录，不当作证据。弃答、预算耗尽、API/执行失败
分别列出并保留在固定分母中；没有有效结论不能当作正确。

API 费用使用 2026-10-03 核对的 [Fireworks 公布价格](https://fireworks.ai/models/fireworks/glm-5p3)：
输入 $1.40、输出 $4.40／百万 tokens。当前汇总保守地把缓存输入按普通输入估算；
这不是账单，且不包含 Modal GPU 费用。实际 GPU 环境和执行时间另外记录。

## 结果

<!-- BEGIN REAL KERNEL RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_106 | reject | [r1](../traces_glm/case_106/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0033 | [r1](../traces_glm/case_106/solo/r1/transcript.md)：reject / 正确；probe 1；$0.1009 | [r1](../traces_glm/case_106/debate/r1/transcript.md)：reject / 正确；probe 3；$0.3190 |
| case_107 | reject | [r1](../traces_glm/case_107/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0027 | [r1](../traces_glm/case_107/solo/r1/transcript.md)：trust / 错误；probe 3；$0.1532 | [r1](../traces_glm/case_107/debate/r1/transcript.md)：trust / 错误；probe 3；$0.2909 |
| case_108 | reject | [r1](../traces_glm/case_108/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0094 | [r1](../traces_glm/case_108/solo/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_108/solo/r2/transcript.md)：trust / 错误；probe 2；$0.1578 | [r1](../traces_glm/case_108/debate/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_108/debate/r2/transcript.md)：trust / 错误；probe 3；$0.4389 |
| case_109 | reject | [r1](../traces_glm/case_109/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0083 | [r1](../traces_glm/case_109/solo/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_109/solo/r2/transcript.md)：reject / 正确；probe 1；$0.2615 | [r1](../traces_glm/case_109/debate/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_109/debate/r2/transcript.md)：— / token 上限；probe 3；$0.4134 |
| case_110 | trust | [r1](../traces_glm/case_110/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0032 | [r1](../traces_glm/case_110/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0964 | [r1](../traces_glm/case_110/debate/r1/transcript.md)：trust / 正确；probe 4；$0.4291 |
| case_111 | trust | [r1](../traces_glm/case_111/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0082 | [r1](../traces_glm/case_111/solo/r1/transcript.md)：trust / 正确；probe 3；$0.2032 | [r1](../traces_glm/case_111/debate/r1/transcript.md)：trust / 正确；probe 3；$0.5007 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 预算耗尽 | 运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 6 | 3 | 3 | 0 | 0 | 0 | 0 | $0.0351 |
| solo | 8 | 4 | 2 | 0 | 0 | 2 | 11 | $0.9730 |
| debate | 8 | 3 | 2 | 0 | 1 | 2 | 19 | $2.3919 |

目标 **18 个 case×arm 槽位**：已有记录 18，缺失 0。共有 22 次尝试；运行失败 4，弃答 0，无最终答案 0，token 上限 1，未完成 0。
全部尝试的已记录 API 费用估算：**$3.4000**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **18 次**。按每组实际评测次数计正确率（预算耗尽仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 3 / 6 | 50.0% |
| solo | 4 / 6 | 66.7% |
| debate | 3 / 6 | 50.0% |

108/109 的四次 r1 工具尝试在 Modal 启动阶段失败，尚未调用模型；其 API 调用数和费用均确认是 0，因此保留故障记录、另用 r2 执行，不作为四次模型答错。下列这些 r1 的 API/runtime 缺项来自进程未启动。

原始提交批次：`run_20261003T045211_881464Z`、`run_20261003T045218_063213Z`、`run_20261003T045703_861566Z`、`run_20261003T045904_045802Z`、`run_20261003T045910_515690Z`、`run_20261003T051611_804489Z`、`run_20261003T051617_071948Z`、`run_20261003T051909_639697Z`、`run_20261003T051915_707230Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。

记录一致性审计问题：

- `case_108/solo/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched
- `case_108/debate/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched
- `case_109/solo/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched
- `case_109/debate/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched

`case_107` 真值修正：初始 trust 已撤回，当前固定标签为 reject。The unchanged contract does not exclude autograd or inference-state tensors; a valid requires_grad leaf triggers a wrapper exception. The previous accept label was an authoring error, not an input-dependent correctness label. 原源码、合同与所有模型 traces 均保持原样；初始资格结果作为历史记录保留。

`case_108` 真值修正：初始 trust 已撤回，当前固定标签为 reject。The original contract did not exclude contiguous lazy-negative PyTorch tensors. Raw pointers ignore that logical flag. Initial case_108 trust qualification omitted this legal input class; old evidence remains intact. A concrete legal counterexample corrects the label to reject. 原源码、合同与所有模型 traces 均保持原样；初始资格结果作为历史记录保留。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 6 例，其中 110–111 是补审后新增的修复实现；单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL RESULTS -->

## 原因分析

**106 的竞争能被直接读源码识别。** 无工具正确指出 STATE/OUT 别名；solo 和
debate 额外执行了合法输入，得到实际位级错误。因此该例增强了可复现证据，
没有拉开最终判断的准确率。

**109 出现了 solo 相对无工具的判断收益。** 无工具把“FP32 partial buffer”误当成
整个归约都使用 FP32。solo 运行多尺寸对照，发现 block 分支 dw 超界而 row 分支
正常；其合法 [512,128] 输入得到 8.94457 倍容差的误差，并定位到先做低精度
归约、后加入 FP32 缓冲的损失。仅保存结果为 FP32 无法补回已丢失的精度。
但无工具只使用 1,092 输出 tokens，solo 使用 7,160 且能多轮思考，因此这里证明
的是本轮“带工具流程”有收益，尚未排除额外思考量的影响。

**109 的 debate 没有完成裁决。** Describer 使用 3,637、Skeptic 4,319、
Experimenter 24,812 输出 tokens，累计正好 32,768；Judge 未发言。实验角色一次
调用使用 24,356 tokens，两个 probe 分别因变量未定义和断言失败而报错；第三个
probe 报告 dw 超界，但下一次模型调用只剩 456 tokens，全部用在 reasoning，
没有登记证据或最终 verdict。它计为预算耗尽，不能由作者读到 stdout 后替模型
追认 reject；也没有给 debate 单独追加预算重跑。

**107、108 是三组共同漏检。** 模型都关注了预期的并行或数值机制，没有覆盖
合同未排除的张量状态/逻辑视图。即使模型高置信度接受、很多测试通过，也不代表
实现满足全部合法域。这两个负结果与作者初始真值错误均保留，未删除或改成成功。
110/111 是补审后新增的修复代码，使用相同合同，三组均接受；这两个对照没有
产生准确率差。最终单次调用 3/6、solo 4/6、debate 3/6（含一次预算耗尽）。

### 逐步 trace 复核：工具究竟增加了什么

这轮没有展示稳定工具优势。109 的轨迹支持实测增加了有用信息，但这不是
仅切换工具开关的因果对照：三组还改变了系统提示、调用轮数和实际思考量。

| Case | 工具前后的可见变化 | 可以支持的结论 |
|---|---|---|
| 106 | solo 在首个 probe 前已指出原地 alias race，随后实测复现 | 工具补充反例，未改变最终分类 |
| 107 | solo 先认为实现正确，23 个普通张量几何配置通过后维持接受 | 关键 Tensor 状态没有被测试；另有下述范围指令冲突 |
| 108 | solo 先认为 FP32/casting 正确，30 个普通数值配置通过后接受 | 没有覆盖逻辑负视图；数值正确不推出整个 wrapper 正确 |
| 109 | solo 起初关注 H16 产品舍入，读取 t9 后首次具体定位 block 内 FP16 归约损失 | 有工具结果推动诊断的直接轨迹证据；探测前未提交 verdict，不能写成已提交 accept 被翻转 |
| 110–111 | 三组均接受；工具组未真正构造修复所处理的 grad/逻辑负视图条件 | 答案正确，但关键修复覆盖没有成立 |

109 的精确证据：[单次思考第 3 行](../traces_glm/case_109/single_call/r1/response_thinking.txt)
把 block 归约当作 FP32；[solo t9 原始输出](../traces_glm/case_109/solo/r2/probes/t9_stdout.txt)
给出合法 [512,128] 反例；[solo 时间线](../traces_glm/case_109/solo/r2/transcript.md)
在读取结果后才定位到 FP16 block sum，并反驳最初的产品舍入假设。

**范围指令存在混杂，107 不能作为干净的工具能力证据。** 公开合同按完整
连续 FP32 张量域评判，作者补审把未排除的 requires_grad 叶子纳入；但工具组
实际收到的 `scope-policy.md` 要求 autograd 未明确要求时视为 unknown/out-of-scope，
并禁止仅凭“not forbidden”扩大范围。Skeptic 角色提示也重复了该限制。
无工具组的简短系统提示没有这套附加政策。这里有合同范围解释的张力；不能把
全部漏检都归于模型不会使用工具。原合同、标签依据和模型记录保持，后续协议
应显式统一 Tensor 属性属于域内还是域外，再比较能力。
参见 [scope policy](../../verifier/agentic/skills/scope-policy.md)、
[Skeptic 指令](../../verifier/agentic/agents/skeptic.py) 和
[107 无工具实际系统提示](../traces_glm/case_107/single_call/r1/system_prompt.txt)。

**探测没有自动形成独立复核。** 110 的 Experimenter 用普通 randn/zeros/flip
推断 contiguous negative view 不存在；Skeptic 和 Judge 继承了这个错误解释。
111 的越界输入也进入了最终“合法边界测试”摘要。现有执行工具检查进程与证据
登记流程，不会自动验证每个输入满足数值域；公共 error_ratios 也只算输出误差。
因此 probe 成功执行、claim 已关闭、多个角色同意，都不能替代证据有效性检查。

**流程本身消耗了预算。** 12 个工具组运行共出现 22 次 `record_claim` 参数错误
（solo 8，debate 14），均漏了 in_scope 必需的 scope_rationale；另有两次描述
登记错误。109 的 24,356-token Experimenter 调用之后只剩 456，既没消费成功
probe，也没进入 Judge。共享预算实现明确不为 Judge 预留额度，见
[llm.py](../../verifier/agentic/llm.py)。这属于流程损失，不能解释成 GPU 没测出错误。

下一步应先做三项小范围校正：统一三组的范围政策；为工具结果附上独立输入
合法性校验并要求复核生成器/参考；为证据整理和最终裁决预留预算、减少登记
错误。然后对 109 在相同 probe 前上下文下做有/无执行结果的小消融，并重复试验。
这些是待验证的改进建议，本次 trace 分析没有新增 API/GPU 调用或改运行代码。

### 人工证据审计

完整记录和正确结论不保证每个 probe 都合格。逐项记录在
`private_data/evidence_audit.json`：

- 109 solo 的脚本实际有 17 个配置，其中 2 个 extreme 输入 RMS 约 0.2475，
  不满足下限 0.25；排除它们后仍有 6 组合法 block 分支反例，reject 仍有依据。
- 108 debate 的部分探测违反 RMS 或 |dy| 上界，不能把这些结果当作合同内验证。
  其中一部分域审计只在 CPU Torch 2.9.1 重建输入，明确不是冻结 Torch 2.8.0/T4
  环境的完整 probe 重放；最终真值另由独立 T4 补审决定。
- 106 solo 的 `torch.equal(ev, ev.clone())` 不能证明输入未修改；实际状态位级
  反例仍合法且有效。不能把该无效不变性检查算成覆盖。
- 109 debate 的最后一个 probe 出现真实失败信号，但没有保存最大 dw 误差的
  最小输入标识，且模型没有完成证据整理和判断；评分保留预算耗尽。
- 110 debate 错误声称“连续张量不可能带逻辑负号”，用普通张量和 flip 测试
  推断符号位分支不可达；没有实际构造 lazy-negative 输入。110 两个工具组也
  都没有检查 requires_grad 输入。其最终 accept 正确不代表这些论证正确。
- 111 debate 有 80 次使用 `x=0.03*randn`，RMS 约 0.03，低于合同下限 0.25；
  solo 的 t12 共 30 组配置，其中 9 组 rms_extreme 低于下限。另一个 debate
  probe 实际执行 160 次却只报告 80 次。两组都没有实际构造逻辑负视图。
  最终关于边界覆盖的陈述不能照抄，必须剔除无效测试并核对计数；完整输入域
  审计见私有记录。

当前结论是小规模机制观察，不是一般能力证明，也没有展示 debate 比 solo 更好。
后续设计应在模型评测前检查整个 wrapper 的输入表示及调用上下文，不能事后通过
收窄合同保住接受标签。

## 从上一批 traces 得到的方向

2026-10-03 结合 traces 和上游源码初筛。前两个方向已进入下方 112–115 小实验，第三个仍为候选。
不能把来源项目本身称为存在合成修改引入的缺陷。
已有简单 GQA 映射（13）、分页 KV（21）、分块 scan（17）及 Split-K/atomic
（22、34、35），下一步应保留真实实现中的组合边界，避免重复显眼的单行错误。

| 优先级 | 候选方向 | 工具可验证的具体性质 | 真实来源入口 |
|---|---|---|---|
| 1 | Packed-sequence / segmented scan 的跨 chunk 状态传递 | 多条独立序列拼接后逐段结果应等于分别执行；移动分段边界至 chunk 前后，检查前一段状态是否泄漏进下一段 | [Mamba state passing](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_state_passing.py)，实际使用 seq_idx 控制跨 chunk 状态 |
| 2 | Paged attention 的页表、滑动窗口和部分 tile | 调换物理页并同步更新页表后逻辑输出不变；窗口边界经过页/tile 中部时与密集参考一致，分别命中实际 dispatch 分支 | [vLLM 历史修复 #30887](https://github.com/vllm-project/vllm/pull/30887)：整 tile 剪枝仍遗漏部分 tile 内窗外 KV block 的遮罩 |
| 3 | Split-K GEMM 与最终 bias/activation 的配合 | 分段部分和合并后满足完整公式，bias/activation 只在合同规定阶段生效；测试非整除 K、多个 split 和连续调用 | [Triton matmul](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul.py)，已有 split scratch 与最终 reduce 路径 |

这些方向的难点应来自“多个局部步骤组合后是否仍正确”。每组保留不同代码的
修复/缺陷版本，合同和真值在评测前固定；输入用于找反例或检查上述性质。
采用公开输入合法性检查，并先验证参考实现。真实项目未明确支持的配置不加入
合同；无效缓存槽允许的内容、Tensor 属性和调用序列也须预先明确。
是否能拉开无工具与 solo、或 solo 与 debate 的差距仍要由小实验决定。

## 边界组合小实验：case_112–115

2026-10-03 预先确定四题和三组，共 12 次模型实验；全量保留，不按胜负筛题。
题目仍在统一的 `triton_eval_cases/case_NN/`，属于专门构造的挑战。
真值针对整个公开合同；任何合法反例都推翻实现，不随输入或随机种子改判。

| Case | 预声明标签 | 来源与机制 |
|---|---|---|
| 112 | accept / trust | Mamba 机制适配：局部递推、跨块状态、逐 token 合并 |
| 113 | reject | 同合同的合成修改：跨块传递取 chunk 起点的段标识，可能串入上一段状态；不是上游历史 bug |
| 114 | reject | vLLM #30887 修复前机制：窗口外 V 未在 P·V 前清零，合法的无效槽 NaN/Inf 可污染结果 |
| 115 | accept / trust | 同合同加入历史修复的窗口 V 遮罩 |

每对共享完整合同、公开独立参考和实际在 `run()` 执行的输入合法性检查。
明确限定普通物化张量；Attention 合同明确有效 KV 必须有限，无效槽可含任意位模式。
私有资格校验另用 NumPy FP64 参考，保留全部计划和 GPU 失败，不向模型提供标签、
来源清单或私有反例。accept 依据源码论证和边界验证，不声称有限测试是形式证明。

三组使用 Fireworks `accounts/fireworks/models/glm-5p3`、low reasoning。
每题每组累计输出上限 32768（包含 reasoning）；single_call 一次请求，工具组单次
上限 8192。solo/debate 均要求 claim 显式提供 scope 字段。debate 在同一总额度内
保留 8192 用于消费已有证据、独立复核和 Judge 裁决（4096 / 1024 / 3072），
收尾禁止新 probe；不因预算耗尽填造 accept/reject。
这是开跑前的协议修正，与 106–111 分批报告；相同输出上限不等于相同实际思考量。
原始请求、响应、usage、完整对话、probe 代码及输出均保存在 `traces_glm/case_NN/`。

协议与来源：[`experiment_boundary.json`](private_data/experiment_boundary.json)、
[`segmented_state_source.json`](private_data/segmented_state_source.json)、
[`paged_attention_source.json`](private_data/paged_attention_source.json)。

独立 T4 资格记录：112 的 18 个计划（38 次调用）全部通过；113 有 10/18
计划失败，包括跨段隔离反例：下一段目标严格为 0，实际为 7.160109，容差 0.002。
Attention 各 54 次调用：114 失败 24 次，115 全部通过，115 最坏误差为容差的
0.0957 倍。完整记录：[r6 Attention](private_data/qualification/r6.json)、
[r7 分段状态](private_data/qualification/r7.json)。

115 另有源码复核提出的 FP16 subnormal 路径补审：45/45 通过，最坏误差为容差的
0.1699 倍。此项在注册后、单次模型调用开始后补充，不冒充预注册覆盖；源码、合同
和标签未变，未调用评测模型。[完整补审](private_data/qualification/supplement_paged_subnormal.json)。

工具容器只接收当前题的公开文件，不挂载其他题、来源、答案或私有资格脚本；
每次记录 `artifact_visibility.json`，核验当前题的可见文件和哈希。

<!-- BEGIN REAL KERNEL BOUNDARY RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_112 | trust | [r1](../traces_glm/case_112/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0147 | [r1](../traces_glm/case_112/solo/r1/transcript.md)：trust / 正确；probe 2；$0.2176 | [r1](../traces_glm/case_112/debate/r1/transcript.md)：trust / 正确；probe 3；$0.2892 |
| case_113 | reject | [r1](../traces_glm/case_113/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0120 | [r1](../traces_glm/case_113/solo/r1/transcript.md)：reject / 正确；probe 3；$0.1837 | [r1](../traces_glm/case_113/debate/r1/transcript.md)：reject / 正确；probe 2；$0.2441 |
| case_114 | reject | [r1](../traces_glm/case_114/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0202 | [r1](../traces_glm/case_114/solo/r1/transcript.md)：reject / 正确；probe 5；$0.2760 | [r1](../traces_glm/case_114/debate/r1/transcript.md)：reject / 正确；probe 7；$0.6061 |
| case_115 | trust | [r1](../traces_glm/case_115/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0166 | [r1](../traces_glm/case_115/solo/r1/transcript.md)：trust / 正确；probe 7；$0.6315 | [r1](../traces_glm/case_115/debate/r1/transcript.md)：trust / 正确；probe 4；$0.6697 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 预算耗尽 | 运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 4 | 3 | 1 | 0 | 0 | 0 | 0 | $0.0636 |
| solo | 4 | 4 | 0 | 0 | 0 | 0 | 17 | $1.3088 |
| debate | 4 | 4 | 0 | 0 | 0 | 0 | 16 | $1.8091 |

目标 **12 个 case×arm 槽位**：已有记录 12，缺失 0。共有 12 次尝试；运行失败 0，弃答 0，无最终答案 0，token 上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$3.1814**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **12 次**。按每组实际评测次数计正确率（预算耗尽仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 3 / 4 | 75.0% |
| solo | 4 / 4 | 100.0% |
| debate | 4 / 4 | 100.0% |
原始提交批次：`run_20261003T073520_679602Z`、`run_20261003T074151_545824Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 4 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL BOUNDARY RESULTS -->

### 本轮证据审计：判断与根因分开

`case_113` 的无工具最终标签正确，但理由误读了 `_combine`：该函数把当前
token 标签与 **chunk 起点前的固定标签**比较，跨段后的全部 token 都会屏蔽旧
carry，并非只屏蔽边界处一个 token。真正的错误在 `_pass_states` 用 chunk
首 token 的身份解释 chunk 末摘要。

- [113 solo](../traces_glm/case_113/solo/r1/transcript.md) 正确定位这个错误。
  t6 因模块变量被整数覆盖而失败；t7 构造了非法首标签，被公开 validator 拒绝。
  t8 修正后包含 7 个合法配置，5 个失败、2 个控制通过；L=65/K=16/边界40
  的输出最大误差为 1.135175，超出约 0.002 的绝对容差。
- [113 debate](../traces_glm/case_113/debate/r1/transcript.md) 的 t8 确有合法
  反例：L=100/K=32/边界50，输出最大误差 0.361973；t9 对齐边界32通过。
  但最终仍把错误归于正确的 `_combine` 遮罩。数值反例支持 reject，不能据此
  声称错误定位也得到独立复核。probe 中 `violation_tokens=[0,64]` 混入了
  坐标维度，实际首个错误 token 为 64，不是 token 0。
- [112 debate](../traces_glm/case_112/debate/r1/transcript.md) 的 t9/t10/t11
  共 8 次合法调用通过，并在最终解释中正确理解遮罩。覆盖集中于 L=97、H=2、
  D=33，不能把这 8 次测试写成整个合同域的证明。
- [112 solo](../traces_glm/case_112/solo/r1/transcript.md) 的 t6/t9 共 547
  次有记录的合法调用通过，包含 530 种不同 shape/chunk 组合；最大输出绝对误差
  5.96e-7。其实际 D 集合是 `{1,8,33,96}`，并未完全执行 claim 所列的宽度。
  最终 trust 正确，但“覆盖完整输入域”和“未修改输入”两点超出了该 probe 的证据。

上述部分 probe 先把 FP64 参考转为 FP32 再比较，报出的误差不是严格 FP64
误差，但距离阈值足以支持这些具体通过/失败判断。112–113 的工具记录中没有输入修改前的
快照，不能声称它们也验证了输入不变性；这项由作者的独立资格测试另行覆盖。

`case_114` 的 [solo trace](../traces_glm/case_114/solo/r1/transcript.md)
提供了更明确的执行纠错过程：首个模型回复倾向代码正确，认为 score 的 `where`
足以排除垃圾槽；读到数值失败后才指出 P·V 的 `0*NaN` 问题。这是中间推理变化，
不是两次正式 verdict，也没有排除多轮推理本身的作用。

- t7 是 Python 代码错误；t8 把部分有效 KV 槽也填成 NaN，被 validator 拒绝。
- t9 修正合法性后执行了 11 个合法配置，其中 6 个产生 NaN；第 12 个配置
  使用合同不支持的 GQA=16，被拒绝，不能写成完整通过了 12 个配置。
- [t10 最小反例](../traces_glm/case_114/solo/r1/probes/t10_probe.py) 使用
  W=1、Q=1、L=64、D=64，64/64 输出为 NaN，参考有限；另一合法配置也得到
  2304 个非有限输出。它足以支持固定 reject 标签。
- t9 用 `torch.equal` 比较含 NaN 的输入导致伪“不变性失败”，随后模型识别问题，
  t11 用字节比较验证普通 example 输入未修改。t10 所谓相同 finite-garbage
  controls 重新生成了有效数据，不属于严格只改变垃圾槽的对照。
- solo 实测了 NaN，未实测 Inf；Inf 的支持来自作者独立资格记录，不能算作
  该模型自行取得的证据。

`case_115` 的 [solo](../traces_glm/case_115/solo/r1/transcript.md) 最初已经
理解 V 的额外屏蔽，最后正确接受。t15 有 14 组合法形状测试；早期越域配置、
忘记传 CUDA 张量和污染有效槽的失败均留在 trace。t20 对 5 组固定有效数据做了
clean/poison 对照，输出数值差为 0，但其 Python 变量遮蔽错误导致所有已使用
物理页内的无效槽均未被填 NaN，只污染了完全未使用的页。代码复核和 CPU 重建
显示其中 W=41/seed3、W=1/seed6 两组仍能在保留 tile 中读到 NaN 过期槽，
确有覆盖修复机制；其余三组没有。不能把它写成“每个无效槽均被验证”，也不能
把数值相等称为做过输出字节比较。

`case_114` 的 [debate](../traces_glm/case_114/debate/r1/transcript.md) 在
任何 GPU probe 前，Skeptic 就指出了 `0*NaN`；Describer 随后明确只能污染
所有 query 都不可达的位置。Experimenter 的
[t13](../traces_glm/case_114/debate/r1/probes/t13_probe.py) 在 L=100/W=5
时仅污染窗口外 logical64 对应的 V，输入通过 validator，64/64 输出为 NaN，
参考没有 NaN。Judge 使用这个合法反例拒绝。因此这是独立源码复核发现问题、
工具复现，不能把 debate 的正确判断全归因于执行工具。

t13 已足够 reject，debate 之后仍修了两次辅助 probe 的形状错误；一次 sweep
因 CUDA generator 错误留下 1836 次失败尝试，修好后又运行了 2142 次有限值
窗口测试。这些工作没有改变已经成立的拒绝结论，是本轮可见的效率问题。

`case_115` 的 [debate](../traces_glm/case_115/debate/r1/transcript.md) 中，
Describer 提醒 tile 向下取整仍会加载过期槽；t14 正确使用全 cache NaN、再填写
全 batch 可达区的方法，6 组通过，最大误差比 0.04298。t19 另有 12 组合法
数值压力测试通过，最大比 0.04283，探测设计比 solo 的 t20 更完整。
但 t16 的 98 组摘要丢弃异常详情，只能写成 98 次尝试、0 个报告的数值违规；
t19 也没有测量权重来证明所谓 dominant-logit 最坏情形。最后复核未指出这些
局限，不能把 coverage 的文字摘要当成严格证明。

本轮结论：四题全保留；114 登记为“观察到工具流程增益”，112/113/115
保留待归普通集合。single_call 3/4、solo 4/4、debate 4/4，未观察到 debate
相对 solo 的准确率收益。相同输出上限不能隔离多轮推理、独立复核与工具执行的
各自贡献；一轮四题也不能证明稳定优势。

12 个模型实验均返回最终判断，原始 66 次 SDK 请求/响应及 33 次 probe 尝试
齐全（probe 尝试不等于合法成功测试）。4 个 debate 的 32 次模型请求都处于
normal 预算阶段，预留收尾未触发，线上结果不能声称验证了“耗尽前救回答案”。
要求显式 scope 字段后仍有 5 次 record_claim 缺少 scope_rationale，另有 3 次
多传 claim_id；接口修正没有消除模型用错工具。这些失败均保留，未重跑选优。

## 上游来源

- [PyTorch 状态内存重叠读写问题 #164701](https://github.com/pytorch/pytorch/issues/164701)
- [Liger-Kernel RMSNorm backward 修复 #950](https://github.com/linkedin/Liger-Kernel/pull/950)
