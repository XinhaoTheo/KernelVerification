# 专门构造的真实 kernel 验证挑战

Case labels use the October 19, 2026 numbering: cases 33–90 were previously 106–163. Cases 163–168 are new October 20 cases; they do not reuse the earlier labels. Private JSON and raw traces retain the earlier IDs. See [case_map.json](../case_map.json).

2026-10-09。该组是**针对验证机制专门准备的研究挑战集**，不是从生产
kernel 随机抽取的代表性样本。公开 kernel 来自真实计算/缺陷机制或历史修复，
适配、人工修改及测试选择均记录；本组结果独立统计，不与旧题库合并为总准确率。

[Tools vs No Tools: case explanations, code snippets and evidence](../SUCCESS_CASES.md).

[Replication of cases 36, 41, 54 and 55](REPLICATION_RESULTS.md). Cases 61–90 and the new October 20 cases 163–168 have their three-arm results below.

## 实验后的保留分组

当前优先比较**带工具验证流程与无工具单次调用**。成功案例按实际观察到的判断差距及合法验证证据保留；暂不把纯工具因果归因或 debate 超过 solo 作为入选门槛。按此口径，**case_41 继续归入成功案例**。原始结果与证据审计仍完整保留。

| 分组 | Case | 后续用途 |
|---|---|---|
| 成功案例：已观察到工具流程增益，单独保留 | **case_36、case_41、case_54、case_55、case_62、case_68、case_74、case_76、case_78、case_81、case_86、case_90、case_166** | 历史入选：36 的 solo、41/54 的 solo/debate 正确拒绝，无工具错误接受；55 的 solo/debate 正确接受，无工具错误拒绝。62 首轮的 solo/debate 找到数值反例，无工具错误接受，但其 r2 三组均正确，区分度未复现；68 首两轮有差距，但新增 r3–r5 中无工具正确 2/3，solo/debate 均正确 3/3；区分度不稳定；74/76 的新增五轮也有波动，78 的新增五轮均复现工具差距；81 仅首轮有差距，新增五轮无工具均正确，未复现；86/90 的首轮差距与固定五轮复测分别报告；166 在首轮体现工具收益，但无工具已在一轮复测答对，仍是有波动的候选，详见本页末尾 |
| 未体现增益，保留待归普通集合 | case_33、34、35、37、38、39、40、42、43–53、56–61、63–67、69–73、75、77、79–80、82–85、87–89、163–165、167–168 | 保留正确/错误 kernel、来源和全部 traces，后续整理进普通集合 |

整理状态登记在 `case_map.json` 的 `curation_status`，并显示于案例和 GLM 索引。当前仅分组保留，未正式迁移普通集合；编号、公开题目路径、固定标签和 traces 均保留。下方 33–38 六例、39–42 四例、43–46 四例、47–48 两例、49–50 两例、51–52 两例、53–54 两例、三轮探索 55–60 六例、61–66 六例、67–72 六例、73–78 六例、79–84 六例及 85–90 六例分别完整统计，不能仅用筛出的成功案例重算成功率。

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

初始计划是两种机制各两个不同实现，`case_33`–`case_36`，共 12 个模型实验槽位。
独立补审发现两个原拟接受版本还有 wrapper 缺陷，因此保留它们并修正错误真值，
另增修复实现 `case_37`–`case_38`。最终为 **6 例 × 3 组 = 18 个模型槽位**。
这两个追加案例是补审后的修复对照，不能说成预先登记的六例随机样本。
每个新实现完成 GPU 资格检查后才进入付费评测；未覆盖的合法输入类导致的资格
遗漏及后续修正明确披露，原始模型输出不改写。

| 编号 | 机制 | 真实来源与适配 |
|---|---|---|
| case_33–case_34 | 状态滑动窗口原地更新，跨 GPU block 重叠读写 | PyTorch issue #164701 的机制；具体移植及修改在私有 provenance 中说明 |
| case_35–case_36 | RMSNorm backward 权重梯度归约精度 | Liger-Kernel PR #950；适配 T4 的 dtype 路径，不称 A100/BF16 原样复现 |
| case_37 | 状态更新的完整修复 | 独立输出、逻辑视图解析和独立 Triton 位级写回；合同与 33–34 相同 |
| case_38 | RMSNorm 的完整修复 | FP32 归约并解析四个输入的逻辑负号；合同与 35–36 相同 |

## 独立资格检查与真值修正

以下是作者侧 T4 验证，不是模型自行获得的成绩。证据保存在
`private_data/qualification/`；私有目录不会提供给模型。

| Case | 当前唯一标签 | 独立依据 |
|---|---|---|
| 33 | reject | 原地并行滑移，201 次调用中 119 次位级错误；row-markers 大形状 64/64 次失败 |
| 34 | reject | 独立输出消除了目标 race，但合法 requires_grad 叶子使 torch.copy_ 报错；补审还覆盖 inference/负视图状态 |
| 35 | reject | FP32 归约通过原 18 组数值检查，但合法 lazy-negative 输入被 raw pointer 错读；10 组视图测试均失败 |
| 36 | reject | 原 18 组有 9 组误差超界；明确梯度反例误差 0.0625，允许 0.002570625，约 24.31 倍 |
| 37 | accept（程序中为 trust） | 分离读写与逻辑符号位处理的源码分析；201 普通调用＋57 张量状态/连续调用全部通过 |
| 38 | accept（程序中为 trust） | FP32 误差界、逻辑视图解析；原 18 组＋10 组视图检查全部通过 |

**34、35 最初的 accept 是作者资格检查遗漏，不是两个同时有效的输入相关标签。**
原合同没有排除带梯度、inference 或逻辑负号标志的普通张量。发现合法反例后，
两例在原合同下统一纠正为 reject；没有把合同改窄，也没有改原实现。旧 r2/r3
资格记录保留为被补审推翻的历史判断，`validation_gpu.json` 保存当前标签与
`label_correction`，明确指向新证据 r4/r5。所有历史模型判断据当前正确真值评分。

例如 35 的合法负视图输入在 M=N=31 时，某个 dw 应为 -23.25，实际为 +23.25；
误差允许值仅 0.0002425。34 的 requires_grad 叶子反例尺寸为 [2,17,33]、K=3，
直接触发原地写回异常。这些额外缺陷来自本轮独立接口适配，不能说成上游原项目
已经被证实存在的同类问题。37/38 保留完整原合同，通过改代码修复。

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
| case_33 | reject | [r1](../traces_glm/case_33/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0033 | [r1](../traces_glm/case_33/solo/r1/transcript.md)：reject / 正确；probe 1；$0.1009 | [r1](../traces_glm/case_33/debate/r1/transcript.md)：reject / 正确；probe 3；$0.3190 |
| case_34 | reject | [r1](../traces_glm/case_34/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0027 | [r1](../traces_glm/case_34/solo/r1/transcript.md)：trust / 错误；probe 3；$0.1532 | [r1](../traces_glm/case_34/debate/r1/transcript.md)：trust / 错误；probe 3；$0.2909 |
| case_35 | reject | [r1](../traces_glm/case_35/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0094 | [r1](../traces_glm/case_35/solo/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_35/solo/r2/transcript.md)：trust / 错误；probe 2；$0.1578 | [r1](../traces_glm/case_35/debate/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_35/debate/r2/transcript.md)：trust / 错误；probe 3；$0.4389 |
| case_36 | reject | [r1](../traces_glm/case_36/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0083 | [r1](../traces_glm/case_36/solo/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_36/solo/r2/transcript.md)：reject / 正确；probe 1；$0.2615 | [r1](../traces_glm/case_36/debate/r1/trace_meta.json)：— / 运行失败；probe 0；$0.0000；审计未通过<br>[r2](../traces_glm/case_36/debate/r2/transcript.md)：— / token 上限；probe 3；$0.4134 |
| case_37 | trust | [r1](../traces_glm/case_37/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0032 | [r1](../traces_glm/case_37/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0964 | [r1](../traces_glm/case_37/debate/r1/transcript.md)：trust / 正确；probe 4；$0.4291 |
| case_38 | trust | [r1](../traces_glm/case_38/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0082 | [r1](../traces_glm/case_38/solo/r1/transcript.md)：trust / 正确；probe 3；$0.2032 | [r1](../traces_glm/case_38/debate/r1/transcript.md)：trust / 正确；probe 3；$0.5007 |

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

35/36 的四次 r1 工具尝试在 Modal 启动阶段失败，尚未调用模型；其 API 调用数和费用均确认是 0，因此保留故障记录、另用 r2 执行，不作为四次模型答错。下列这些 r1 的 API/runtime 缺项来自进程未启动。

原始提交批次：`run_20261003T045211_881464Z`、`run_20261003T045218_063213Z`、`run_20261003T045703_861566Z`、`run_20261003T045904_045802Z`、`run_20261003T045910_515690Z`、`run_20261003T051611_804489Z`、`run_20261003T051617_071948Z`、`run_20261003T051909_639697Z`、`run_20261003T051915_707230Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。

记录一致性审计问题：

- `case_35/solo/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched
- `case_35/debate/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched
- `case_36/solo/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched
- `case_36/debate/r1`：raw API capture incomplete; runtime verifier fingerprint missing or mismatched

`case_34` 真值修正：初始 trust 已撤回，当前固定标签为 reject。The unchanged contract does not exclude autograd or inference-state tensors; a valid requires_grad leaf triggers a wrapper exception. The previous accept label was an authoring error, not an input-dependent correctness label. 原源码、合同与所有模型 traces 均保持原样；初始资格结果作为历史记录保留。

`case_35` 真值修正：初始 trust 已撤回，当前固定标签为 reject。The original contract did not exclude contiguous lazy-negative PyTorch tensors. Raw pointers ignore that logical flag. Initial case_35 trust qualification omitted this legal input class; old evidence remains intact. A concrete legal counterexample corrects the label to reject. 原源码、合同与所有模型 traces 均保持原样；初始资格结果作为历史记录保留。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 6 例，其中 37–38 是补审后新增的修复实现；单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL RESULTS -->

## 原因分析

**33 的竞争能被直接读源码识别。** 无工具正确指出 STATE/OUT 别名；solo 和
debate 额外执行了合法输入，得到实际位级错误。因此该例增强了可复现证据，
没有拉开最终判断的准确率。

**36 出现了 solo 相对无工具的判断收益。** 无工具把“FP32 partial buffer”误当成
整个归约都使用 FP32。solo 运行多尺寸对照，发现 block 分支 dw 超界而 row 分支
正常；其合法 [512,128] 输入得到 8.94457 倍容差的误差，并定位到先做低精度
归约、后加入 FP32 缓冲的损失。仅保存结果为 FP32 无法补回已丢失的精度。
但无工具只使用 1,092 输出 tokens，solo 使用 7,160 且能多轮思考，因此这里证明
的是本轮“带工具流程”有收益，尚未排除额外思考量的影响。

**36 的 debate 没有完成裁决。** Describer 使用 3,637、Skeptic 4,319、
Experimenter 24,812 输出 tokens，累计正好 32,768；Judge 未发言。实验角色一次
调用使用 24,356 tokens，两个 probe 分别因变量未定义和断言失败而报错；第三个
probe 报告 dw 超界，但下一次模型调用只剩 456 tokens，全部用在 reasoning，
没有登记证据或最终 verdict。它计为预算耗尽，不能由作者读到 stdout 后替模型
追认 reject；也没有给 debate 单独追加预算重跑。

**34、35 是三组共同漏检。** 模型都关注了预期的并行或数值机制，没有覆盖
合同未排除的张量状态/逻辑视图。即使模型高置信度接受、很多测试通过，也不代表
实现满足全部合法域。这两个负结果与作者初始真值错误均保留，未删除或改成成功。
37/38 是补审后新增的修复代码，使用相同合同，三组均接受；这两个对照没有
产生准确率差。最终单次调用 3/6、solo 4/6、debate 3/6（含一次预算耗尽）。

### 逐步 trace 复核：工具究竟增加了什么

这轮没有展示稳定工具优势。36 的轨迹支持实测增加了有用信息，但这不是
仅切换工具开关的因果对照：三组还改变了系统提示、调用轮数和实际思考量。

| Case | 工具前后的可见变化 | 可以支持的结论 |
|---|---|---|
| 33 | solo 在首个 probe 前已指出原地 alias race，随后实测复现 | 工具补充反例，未改变最终分类 |
| 34 | solo 先认为实现正确，23 个普通张量几何配置通过后维持接受 | 关键 Tensor 状态没有被测试；另有下述范围指令冲突 |
| 35 | solo 先认为 FP32/casting 正确，30 个普通数值配置通过后接受 | 没有覆盖逻辑负视图；数值正确不推出整个 wrapper 正确 |
| 36 | solo 起初关注 H16 产品舍入，读取 t9 后首次具体定位 block 内 FP16 归约损失 | 有工具结果推动诊断的直接轨迹证据；探测前未提交 verdict，不能写成已提交 accept 被翻转 |
| 37–38 | 三组均接受；工具组未真正构造修复所处理的 grad/逻辑负视图条件 | 答案正确，但关键修复覆盖没有成立 |

36 的精确证据：[单次思考第 3 行](../traces_glm/case_36/single_call/r1/response_thinking.txt)
把 block 归约当作 FP32；[solo t9 原始输出](../traces_glm/case_36/solo/r2/probes/t9_stdout.txt)
给出合法 [512,128] 反例；[solo 时间线](../traces_glm/case_36/solo/r2/transcript.md)
在读取结果后才定位到 FP16 block sum，并反驳最初的产品舍入假设。

**范围指令存在混杂，34 不能作为干净的工具能力证据。** 公开合同按完整
连续 FP32 张量域评判，作者补审把未排除的 requires_grad 叶子纳入；但工具组
实际收到的 `scope-policy.md` 要求 autograd 未明确要求时视为 unknown/out-of-scope，
并禁止仅凭“not forbidden”扩大范围。Skeptic 角色提示也重复了该限制。
无工具组的简短系统提示没有这套附加政策。这里有合同范围解释的张力；不能把
全部漏检都归于模型不会使用工具。原合同、标签依据和模型记录保持，后续协议
应显式统一 Tensor 属性属于域内还是域外，再比较能力。
参见 [scope policy](../../verifier/agentic/skills/scope-policy.md)、
[Skeptic 指令](../../verifier/agentic/agents/skeptic.py) 和
[34 无工具实际系统提示](../traces_glm/case_34/single_call/r1/system_prompt.txt)。

**探测没有自动形成独立复核。** 37 的 Experimenter 用普通 randn/zeros/flip
推断 contiguous negative view 不存在；Skeptic 和 Judge 继承了这个错误解释。
38 的越界输入也进入了最终“合法边界测试”摘要。现有执行工具检查进程与证据
登记流程，不会自动验证每个输入满足数值域；公共 error_ratios 也只算输出误差。
因此 probe 成功执行、claim 已关闭、多个角色同意，都不能替代证据有效性检查。

**流程本身消耗了预算。** 12 个工具组运行共出现 22 次 `record_claim` 参数错误
（solo 8，debate 14），均漏了 in_scope 必需的 scope_rationale；另有两次描述
登记错误。36 的 24,356-token Experimenter 调用之后只剩 456，既没消费成功
probe，也没进入 Judge。共享预算实现明确不为 Judge 预留额度，见
[llm.py](../../verifier/agentic/llm.py)。这属于流程损失，不能解释成 GPU 没测出错误。

下一步应先做三项小范围校正：统一三组的范围政策；为工具结果附上独立输入
合法性校验并要求复核生成器/参考；为证据整理和最终裁决预留预算、减少登记
错误。然后对 36 在相同 probe 前上下文下做有/无执行结果的小消融，并重复试验。
这些是待验证的改进建议，本次 trace 分析没有新增 API/GPU 调用或改运行代码。

### 人工证据审计

完整记录和正确结论不保证每个 probe 都合格。逐项记录在
`private_data/evidence_audit.json`：

- 36 solo 的脚本实际有 17 个配置，其中 2 个 extreme 输入 RMS 约 0.2475，
  不满足下限 0.25；排除它们后仍有 6 组合法 block 分支反例，reject 仍有依据。
- 35 debate 的部分探测违反 RMS 或 |dy| 上界，不能把这些结果当作合同内验证。
  其中一部分域审计只在 CPU Torch 2.9.1 重建输入，明确不是冻结 Torch 2.8.0/T4
  环境的完整 probe 重放；最终真值另由独立 T4 补审决定。
- 33 solo 的 `torch.equal(ev, ev.clone())` 不能证明输入未修改；实际状态位级
  反例仍合法且有效。不能把该无效不变性检查算成覆盖。
- 36 debate 的最后一个 probe 出现真实失败信号，但没有保存最大 dw 误差的
  最小输入标识，且模型没有完成证据整理和判断；评分保留预算耗尽。
- 37 debate 错误声称“连续张量不可能带逻辑负号”，用普通张量和 flip 测试
  推断符号位分支不可达；没有实际构造 lazy-negative 输入。37 两个工具组也
  都没有检查 requires_grad 输入。其最终 accept 正确不代表这些论证正确。
- 38 debate 有 80 次使用 `x=0.03*randn`，RMS 约 0.03，低于合同下限 0.25；
  solo 的 t12 共 30 组配置，其中 9 组 rms_extreme 低于下限。另一个 debate
  probe 实际执行 160 次却只报告 80 次。两组都没有实际构造逻辑负视图。
  最终关于边界覆盖的陈述不能照抄，必须剔除无效测试并核对计数；完整输入域
  审计见私有记录。

当前结论是小规模机制观察，不是一般能力证明，也没有展示 debate 比 solo 更好。
后续设计应在模型评测前检查整个 wrapper 的输入表示及调用上下文，不能事后通过
收窄合同保住接受标签。

## 从上一批 traces 得到的方向

2026-10-03 结合 traces 和上游源码初筛。前两个方向已进入下方 39–42 小实验，第三个仍为候选。
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

## 边界组合小实验：case_39–case_42

2026-10-03 预先确定四题和三组，共 12 次模型实验；全量保留，不按胜负筛题。
题目仍在统一的 `triton_eval_cases/case_NN/`，属于专门构造的挑战。
真值针对整个公开合同；任何合法反例都推翻实现，不随输入或随机种子改判。

| Case | 预声明标签 | 来源与机制 |
|---|---|---|
| 39 | accept / trust | Mamba 机制适配：局部递推、跨块状态、逐 token 合并 |
| 40 | reject | 同合同的合成修改：跨块传递取 chunk 起点的段标识，可能串入上一段状态；不是上游历史 bug |
| 41 | reject | vLLM #30887 修复前机制：窗口外 V 未在 P·V 前清零，合法的无效槽 NaN/Inf 可污染结果 |
| 42 | accept / trust | 同合同加入历史修复的窗口 V 遮罩 |

每对共享完整合同、公开独立参考和实际在 `run()` 执行的输入合法性检查。
明确限定普通物化张量；Attention 合同明确有效 KV 必须有限，无效槽可含任意位模式。
私有资格校验另用 NumPy FP64 参考，保留全部计划和 GPU 失败，不向模型提供标签、
来源清单或私有反例。accept 依据源码论证和边界验证，不声称有限测试是形式证明。

三组使用 Fireworks `accounts/fireworks/models/glm-5p3`、low reasoning。
每题每组累计输出上限 32768（包含 reasoning）；single_call 一次请求，工具组单次
上限 8192。solo/debate 均要求 claim 显式提供 scope 字段。debate 在同一总额度内
保留 8192 用于消费已有证据、独立复核和 Judge 裁决（4096 / 1024 / 3072），
收尾禁止新 probe；不因预算耗尽填造 accept/reject。
这是开跑前的协议修正，与 33–38 分批报告；相同输出上限不等于相同实际思考量。
原始请求、响应、usage、完整对话、probe 代码及输出均保存在 `traces_glm/case_NN/`。

协议与来源：[`experiment_boundary.json`](private_data/experiment_boundary.json)、
[`segmented_state_source.json`](private_data/segmented_state_source.json)、
[`paged_attention_source.json`](private_data/paged_attention_source.json)。

独立 T4 资格记录：39 的 18 个计划（38 次调用）全部通过；40 有 10/18
计划失败，包括跨段隔离反例：下一段目标严格为 0，实际为 7.160109，容差 0.002。
Attention 各 54 次调用：41 失败 24 次，42 全部通过，42 最坏误差为容差的
0.0957 倍。完整记录：[r6 Attention](private_data/qualification/r6.json)、
[r7 分段状态](private_data/qualification/r7.json)。

42 另有源码复核提出的 FP16 subnormal 路径补审：45/45 通过，最坏误差为容差的
0.1699 倍。此项在注册后、单次模型调用开始后补充，不冒充预注册覆盖；源码、合同
和标签未变，未调用评测模型。[完整补审](private_data/qualification/supplement_paged_subnormal.json)。

工具容器只接收当前题的公开文件，不挂载其他题、来源、答案或私有资格脚本；
每次记录 `artifact_visibility.json`，核验当前题的可见文件和哈希。

<!-- BEGIN REAL KERNEL BOUNDARY RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_39 | trust | [r1](../traces_glm/case_39/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0147 | [r1](../traces_glm/case_39/solo/r1/transcript.md)：trust / 正确；probe 2；$0.2176 | [r1](../traces_glm/case_39/debate/r1/transcript.md)：trust / 正确；probe 3；$0.2892 |
| case_40 | reject | [r1](../traces_glm/case_40/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0120 | [r1](../traces_glm/case_40/solo/r1/transcript.md)：reject / 正确；probe 3；$0.1837 | [r1](../traces_glm/case_40/debate/r1/transcript.md)：reject / 正确；probe 2；$0.2441 |
| case_41 | reject | [r1](../traces_glm/case_41/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0202 | [r1](../traces_glm/case_41/solo/r1/transcript.md)：reject / 正确；probe 5；$0.2760 | [r1](../traces_glm/case_41/debate/r1/transcript.md)：reject / 正确；probe 7；$0.6061 |
| case_42 | trust | [r1](../traces_glm/case_42/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0166 | [r1](../traces_glm/case_42/solo/r1/transcript.md)：trust / 正确；probe 7；$0.6315 | [r1](../traces_glm/case_42/debate/r1/transcript.md)：trust / 正确；probe 4；$0.6697 |

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

`case_40` 的无工具最终标签正确，但理由误读了 `_combine`：该函数把当前
token 标签与 **chunk 起点前的固定标签**比较，跨段后的全部 token 都会屏蔽旧
carry，并非只屏蔽边界处一个 token。真正的错误在 `_pass_states` 用 chunk
首 token 的身份解释 chunk 末摘要。

- [40 solo](../traces_glm/case_40/solo/r1/transcript.md) 正确定位这个错误。
  t6 因模块变量被整数覆盖而失败；t7 构造了非法首标签，被公开 validator 拒绝。
  t8 修正后包含 7 个合法配置，5 个失败、2 个控制通过；L=65/K=16/边界40
  的输出最大误差为 1.135175，超出约 0.002 的绝对容差。
- [40 debate](../traces_glm/case_40/debate/r1/transcript.md) 的 t8 确有合法
  反例：L=100/K=32/边界50，输出最大误差 0.361973；t9 对齐边界32通过。
  但最终仍把错误归于正确的 `_combine` 遮罩。数值反例支持 reject，不能据此
  声称错误定位也得到独立复核。probe 中 `violation_tokens=[0,64]` 混入了
  坐标维度，实际首个错误 token 为 64，不是 token 0。
- [39 debate](../traces_glm/case_39/debate/r1/transcript.md) 的 t9/t10/t11
  共 8 次合法调用通过，并在最终解释中正确理解遮罩。覆盖集中于 L=97、H=2、
  D=33，不能把这 8 次测试写成整个合同域的证明。
- [39 solo](../traces_glm/case_39/solo/r1/transcript.md) 的 t6/t9 共 547
  次有记录的合法调用通过，包含 530 种不同 shape/chunk 组合；最大输出绝对误差
  5.96e-7。其实际 D 集合是 `{1,8,33,96}`，并未完全执行 claim 所列的宽度。
  最终 trust 正确，但“覆盖完整输入域”和“未修改输入”两点超出了该 probe 的证据。

上述部分 probe 先把 FP64 参考转为 FP32 再比较，报出的误差不是严格 FP64
误差，但距离阈值足以支持这些具体通过/失败判断。39–40 的工具记录中没有输入修改前的
快照，不能声称它们也验证了输入不变性；这项由作者的独立资格测试另行覆盖。

`case_41` 的 [solo trace](../traces_glm/case_41/solo/r1/transcript.md)
提供了更明确的执行纠错过程：首个模型回复倾向代码正确，认为 score 的 `where`
足以排除垃圾槽；读到数值失败后才指出 P·V 的 `0*NaN` 问题。这是中间推理变化，
不是两次正式 verdict，也没有排除多轮推理本身的作用。

- t7 是 Python 代码错误；t8 把部分有效 KV 槽也填成 NaN，被 validator 拒绝。
- t9 修正合法性后执行了 11 个合法配置，其中 6 个产生 NaN；第 12 个配置
  使用合同不支持的 GQA=16，被拒绝，不能写成完整通过了 12 个配置。
- [t10 最小反例](../traces_glm/case_41/solo/r1/probes/t10_probe.py) 使用
  W=1、Q=1、L=64、D=64，64/64 输出为 NaN，参考有限；另一合法配置也得到
  2304 个非有限输出。它足以支持固定 reject 标签。
- t9 用 `torch.equal` 比较含 NaN 的输入导致伪“不变性失败”，随后模型识别问题，
  t11 用字节比较验证普通 example 输入未修改。t10 所谓相同 finite-garbage
  controls 重新生成了有效数据，不属于严格只改变垃圾槽的对照。
- solo 实测了 NaN，未实测 Inf；Inf 的支持来自作者独立资格记录，不能算作
  该模型自行取得的证据。

`case_42` 的 [solo](../traces_glm/case_42/solo/r1/transcript.md) 最初已经
理解 V 的额外屏蔽，最后正确接受。t15 有 14 组合法形状测试；早期越域配置、
忘记传 CUDA 张量和污染有效槽的失败均留在 trace。t20 对 5 组固定有效数据做了
clean/poison 对照，输出数值差为 0，但其 Python 变量遮蔽错误导致所有已使用
物理页内的无效槽均未被填 NaN，只污染了完全未使用的页。代码复核和 CPU 重建
显示其中 W=41/seed3、W=1/seed6 两组仍能在保留 tile 中读到 NaN 过期槽，
确有覆盖修复机制；其余三组没有。不能把它写成“每个无效槽均被验证”，也不能
把数值相等称为做过输出字节比较。

`case_41` 的 [debate](../traces_glm/case_41/debate/r1/transcript.md) 在
任何 GPU probe 前，Skeptic 就指出了 `0*NaN`；Describer 随后明确只能污染
所有 query 都不可达的位置。Experimenter 的
[t13](../traces_glm/case_41/debate/r1/probes/t13_probe.py) 在 L=100/W=5
时仅污染窗口外 logical64 对应的 V，输入通过 validator，64/64 输出为 NaN，
参考没有 NaN。Judge 使用这个合法反例拒绝。因此这是独立源码复核发现问题、
工具复现，不能把 debate 的正确判断全归因于执行工具。

t13 已足够 reject，debate 之后仍修了两次辅助 probe 的形状错误；一次 sweep
因 CUDA generator 错误留下 1836 次失败尝试，修好后又运行了 2142 次有限值
窗口测试。这些工作没有改变已经成立的拒绝结论，是本轮可见的效率问题。

`case_42` 的 [debate](../traces_glm/case_42/debate/r1/transcript.md) 中，
Describer 提醒 tile 向下取整仍会加载过期槽；t14 正确使用全 cache NaN、再填写
全 batch 可达区的方法，6 组通过，最大误差比 0.04298。t19 另有 12 组合法
数值压力测试通过，最大比 0.04283，探测设计比 solo 的 t20 更完整。
但 t16 的 98 组摘要丢弃异常详情，只能写成 98 次尝试、0 个报告的数值违规；
t19 也没有测量权重来证明所谓 dominant-logit 最坏情形。最后复核未指出这些
局限，不能把 coverage 的文字摘要当成严格证明。

本轮结论：四题全保留；41 登记为“观察到工具流程增益”，39/40/42
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


## Split-K 与 MoE：case_43–case_46

2026-10-03 新一轮四例小实验，主要目标仍是比较**带工具流程与无工具单次调用**。
使用真实工程机制的 T4 适配及明确标注的人工修改；不称为上游已确认历史 bug。
每对共享完整公开合同，以不同实现构成固定 accept/reject 对照。输入只用于检验
实现是否违反合同，不按输入选择标签。所有四例和三组尝试均保留，未体现增益的
案例后续归入普通候选，不依据模型答错与否修改本轮源码或阈值。

| Case | 预登记标签 | 机制与工具要检验的问题 |
|---|---|---|
| 43 | reject | Split-K GEMM：FP32 累加后经 FP16 中间缓冲，有限 partial 的舍入在合并时造成超界 |
| 44 | accept | 同一 GEMM 合同，FP32 中间缓冲；以误差界与独立数值检查支持 |
| 45 | accept | MoE 按 expert 分桶、padding、grouped GEMM、按 route 回写与 top-k 加权 |
| 46 | reject | 同一 MoE 合同，grouped GEMM 尾组的调度映射遗漏输出 tile |

来源、合成修改、完整合同依据、作者独立资格记录均保存在 `private_data/`，不会
发送给被测模型。公开题目仍为 `triton_eval_cases/case_43`–`case_46`。
本批状态及冻结记录见 `private_data/experiment_splitk_moe.json`。

沿用 39–42 的协议：Fireworks GLM-5p3，reasoning effort low；三组各 32768
总输出 tokens，工具调用单轮上限 8192，debate 保留收尾预算并要求 claim 范围依据。
两工具组均使用 T4，容器只看到当前题公开文件，完整记录 API、probe、失败及 usage。
不改 verifier 流程以针对新题取得优势；正确率、执行证据和推理根因分别审计。

### 本批独立验收

Split-K 来源为 Triton `7d85777c…` 的 split-K scratch／最终归约机制；MoE 来源为
vLLM `8c1557a7…` 的 grouped GEMM 与路由机制。前者为独立的小型实现，后者提取
并适配一层专家投影；均有独立参考和许可证，不是完整生产 kernel 原样测评。

| Case | 固定标签 | 模型调用前的独立 T4 证据 |
|---|---|---|
| 43 | reject | 33 组计划中 1 组失败：目标 0.0078125，实际 0.03125，误差 2.3419 倍容差；全部有限 |
| 44 | accept | 33 组、70 次调用通过；最坏误差 0.000223 倍容差；最大 subnormal×normal 检查通过 |
| 45 | accept | 45 次调用通过；含路由顺序／专家重标、非零 storage offset 和输入保持；最坏误差 0.000118 倍容差 |
| 46 | reject | 45 次调用有 20 次失败；例如合法 [M=17,E=5,Q=1,K=33,N=33] 下 [4,32] 应为8.25却输出0，误差708.15倍容差 |

记录在 `private_data/qualification/r8.json` 与 `r9.json`。接受版另有源码检查、
数值误差界及 MoE 有限 tile-grid 的双射枚举支持；这些不是全输入域机器证明。
所有公开文件先冻结再付费评测；四例无需在 GPU 验收失败后修改实现或标签。

<!-- BEGIN REAL KERNEL SPLITK MOE RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_43 | reject | [r1](../traces_glm/case_43/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0060 | [r1](../traces_glm/case_43/solo/r1/transcript.md)：reject / 正确；probe 3；$0.1943 | [r1](../traces_glm/case_43/debate/r1/transcript.md)：needs_more_evidence / 弃答；probe 6；$0.8744 |
| case_44 | trust | [r1](../traces_glm/case_44/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0056 | [r1](../traces_glm/case_44/solo/r1/transcript.md)：trust / 正确；probe 2；$0.1118 | [r1](../traces_glm/case_44/debate/r1/transcript.md)：trust / 正确；probe 4；$0.3241 |
| case_45 | trust | [r1](../traces_glm/case_45/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0089 | [r1](../traces_glm/case_45/solo/r1/transcript.md)：trust / 正确；probe 3；$0.2582 | [r1](../traces_glm/case_45/debate/r1/transcript.md)：trust / 正确；probe 4；$0.2901 |
| case_46 | reject | [r1](../traces_glm/case_46/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0102 | [r1](../traces_glm/case_46/solo/r1/transcript.md)：reject / 正确；probe 3；$0.2030 | [r1](../traces_glm/case_46/debate/r1/transcript.md)：reject / 正确；probe 1；$0.2276 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 预算耗尽 | 运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 4 | 4 | 0 | 0 | 0 | 0 | 0 | $0.0306 |
| solo | 4 | 4 | 0 | 0 | 0 | 0 | 11 | $0.7673 |
| debate | 4 | 3 | 0 | 1 | 0 | 0 | 15 | $1.7162 |

目标 **12 个 case×arm 槽位**：已有记录 12，缺失 0。共有 12 次尝试；运行失败 0，弃答 1，无最终答案 0，token 上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$2.5141**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **12 次**。按每组实际评测次数计正确率（预算耗尽仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 4 / 4 | 100.0% |
| solo | 4 / 4 | 100.0% |
| debate | 3 / 4 | 75.0% |
原始提交批次：`run_20261003T180932_122767Z`、`run_20261003T180932_347132Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 4 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL SPLITK MOE RESULTS -->


### 本批逐项 trace 审计

无工具四例全部答对。因此，本批不能归入“工具流程提高最终判断准确率”的成功
案例；具体工具证据仍逐项保留。以下审计区分模型 verdict、合法运行证据以及
模型对测试覆盖的解释，不用模型置信度作为证明。

**43：solo 增加了可复现证据，debate 的反例生成失败。** 无工具看到 FP16
scratch 就拒绝，方向正确，但没有给出可复现数值例；其“合同禁止中间舍入”
解释不准确，合同允许近似，只要求最终误差合格。它还把只可能在 split=1
出现的 partial 上限128用于 half-scratch 分支；split≥2时实际最大为64。

solo 在执行前已怀疑这个缺陷，并非看输出后从 accept 翻转。第一份
[t6](../traces_glm/case_43/solo/r1/probes/t6_probe.py) 使用±0.5，partial 恰好可表示，
360 次合法调用均通过；模型随后指出测试没有触发舍入，改用
[t9](../traces_glm/case_43/solo/r1/probes/t9_probe.py) 的非整齐正负分片。
180 次合法调用出现 **40 个违规输出元素**，不是40个失败配置。记录的反例为
实际 -0.390625、FP64参考 -0.40583811516989954，误差0.0152131超过容差0.0104058。
另有一次未 finalize 前重跑被流程挡住，未实际执行。这体现工具帮助检验和改进
反例设计，未体现最终标签准确率提升。

debate 的首个舍入探测也用了恰好可表示的±32 partial。后续先出现CPU/CUDA混用
（t15）、再出现矩阵形状错误（t20）。最终
[t22](../traces_glm/case_43/debate/r1/probes/t22_probe.py) 设三个正段，每段最小和16，
希望最后一段抵消；但最后段最大绝对和32，而前三段合计至少48。其需要的B≤-0.75
违反|B|≤0.5，故200个候选全部被过滤，**candidate kernel调用数为0**。
这不是随机搜索运气差。脚本还有未走到的partial诊断形状错误，不能算已发生的错误。
Describer/Skeptic没有纠正不可行构造；重新打开claim的调度实际生效，但实验仍失败。
最终 `needs_more_evidence` 是有剩余预算时的主动弃答，不是token耗尽，不计正确。

**44：正确对照的通过测试不等于全域证明。** 无工具和两个工具组都接受。
工具组的实测覆盖与最终“风险全部反驳”的措辞须区分。例如 debate 的±0.5
“最坏精度测试”依然只产生精确0.25倍数，并没有检验FP32最坏舍入；测试通过
只能结合源码与误差界支持接受，不能说从有限随机测试证明所有合法输入。

**45：solo 的76次尝试中只有65次合法运行。** 11次分别因route权重shape错误
或未归一化超过合同上限而被公开validator拒绝；不能把异常当成kernel缺陷，
也不能记为76组合法覆盖。合法样本最坏绝对误差约1.0628e-6。模型在probe前已经
推导正确的尾组映射，工具起确认作用。它没有检查输入保持、输出不别名、非零
storage offset或subnormal；这些范围不能由普通随机通过替代。


44 solo 实际有1104次合法调用，但tiny测试让双方都约1e-7，贡献远小于容差，
不是敏感的subnormal×normal测试；`torch.equal`也不能支持其“bitwise-equal”措辞。
44 debate有331次合法调用，另一次Python语法错误未执行；offset与敏感subnormal
由作者独立资格检查覆盖，不是评测agent自行覆盖的成绩。

**45 debate：原始补测通过，但证据状态没有同步。** t10在生成ids时IndexError，
未调用kernel；t15修复后5/6/7-block三个合法配置均通过，配合t9、t11共5次合法运行。
但[t15](../traces_glm/case_45/debate/r1/probes/t15_probe.py) 没有被 finalize 到
[claims.json](../traces_glm/case_45/debate/r1/claims.json)，c2最终仍为 `inconclusive`，
只附着失败的t10；Skeptic和Judge却称三条风险均已反驳。当前 verdict guard 只挡住
`open`，不会挡住 `inconclusive`，因此允许提交trust。这是证据状态与说明不一致，
不是kernel标签错误。原记录保留，本批不修改流程或重跑。其最大幅值测试全用0.5、
路由权重0.25，恰为精确二进制算术，不能声称验证了最坏舍入。

**46：工具验证了静态分析已经找到的错误。** 无工具直接对比并手算了
`GROUP_M`与`group_size_m`的尾组映射，两工具组也在第一次probe前发现同一问题。
solo前两次尝试分别因ids dtype错误和Python重复关键字失败，第三次
[t8](../traces_glm/case_46/solo/r1/probes/t8_probe.py) 的9个合法配置有5个失败、
4个完整组对照通过，最大绝对误差0.925845。debate的
[t7](../traces_glm/case_46/debate/r1/probes/t7_probe.py) 用5个专家各16条route，
M80/K32/N96/Q1，966个输出违反容差，最大绝对误差0.689916，受影响列为32–95。
这些都是agent自行构造的合法反例，足以拒绝固定实现；没有从无工具的错误结论
翻转为正确结论。工具增加了直接证据，未增加本轮分类准确率。

### 本批结论与保留

本批**没有新增工具增益成功案例**：single_call 4/4，solo 4/4，debate 3/4加一次
弃答。四例均登记为 `ordinary_candidate`，成功集合仍为case_36、case_41。
61次SDK请求均有完整原始记录，26次probe工具尝试对应24份脚本；被流程挡住的
尝试、Python错误、非法样本和零候选的“成功退出”都保留，不能当作26次成功实验。
API估算$2.514113，不含GPU，非实际账单；没有预算耗尽、API失败或事后重跑。

没有区分度的主要原因是**源码中的静态信号仍然太明显**：43暴露FP16 scratch，
46是模型熟悉的grouped-GEMM尾组公式，两者都能用较短源码分析给出正确标签。
增加真实工程步骤没有自动增加必须执行才能解决的难度。后续更值得研究的是
公开合同下的编译、线程布局或多阶段归约语义边界，并先验证静态分析到底缺少哪项
信息；不靠改输入标签、隐藏合同或堆代码量制造难度。本轮不继续扩成十例。


## INT4 完整路径：case_47–case_48

2026-10-03 新的两例小实验，目标仍是**带工具流程相对无工具单次调用的最终判定
增益**。保留真实 INT4 解包、分组参数、重排与 GEMM 的必要阶段，检查阶段之间
的数据格式约定，不通过额外混淆代码制造难度。正确／错误版本共享完整公开合同；
预登记 case_47 为 accept、case_48 为 reject，必须独立验收后才启动模型。

标准答案由**已存储的 packed 数值和公开解码格式**定义，不用量化前的原矩阵作为
标准答案，也不按输入是否通过切换标签。任何格式适配或人工 mutation 均在
`private_data/int4_source.json` 记录；私有来源和资格证据不会进入被测环境。

公开题目位于 `triton_eval_cases/case_47`、`case_48`；资格程序为
`eval_scripts/real_kernel_challenges/int4_pair.py`。本批协议与状态记录在
`private_data/experiment_int4_pipeline.json`，沿用上一批 GLM-5p3、三组预算和
当前题隔离机制，完整记录原始 API、工具代码、输出、失败和费用。

<!-- BEGIN REAL KERNEL INT4 PIPELINE RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_47 | trust | [r1](../traces_glm/case_47/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0268 | [r1](../traces_glm/case_47/solo/r1/transcript.md)：trust / 正确；probe 4；$0.2833 | [r1](../traces_glm/case_47/debate/r1/transcript.md)：trust / 正确；probe 5；$0.4550 |
| case_48 | reject | [r1](../traces_glm/case_48/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0937 | [r1](../traces_glm/case_48/solo/r1/transcript.md)：reject / 正确；probe 4；$0.3075 | [r1](../traces_glm/case_48/debate/r1/transcript.md)：reject / 正确；probe 5；$0.4256 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 预算耗尽 | 运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.1204 |
| solo | 2 | 2 | 0 | 0 | 0 | 0 | 8 | $0.5908 |
| debate | 2 | 2 | 0 | 0 | 0 | 0 | 10 | $0.8805 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；运行失败 0，弃答 0，无最终答案 0，token 上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$1.5918**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（预算耗尽仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 2 / 2 | 100.0% |
| solo | 2 / 2 | 100.0% |
| debate | 2 / 2 | 100.0% |
原始提交批次：`run_20261003T190508_997393Z`、`run_20261003T190509_138229Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL INT4 PIPELINE RESULTS -->

### 构造与独立验收

采用 AutoGPTQ 固定 commit `9f7d37072917ab3a7545835f23e808294a542153` 中的
GPTQ 存储约定与 Marlin 16×64 tile 排列，适配为 T4 上独立的 Triton 流程。
原版 Marlin 的 Ampere CUDA 内核不支持 T4；这里保留其布局机制，并明确标注
小尺寸、尾部、非对称 zero point 等适配，不能称作原版生产内核或历史 bug 复现。
两份许可证及逐文件来源保存在公开 `LICENSE` 和私有来源记录中。

case_47 的 repack 采用原始八通道 interleave；case_48 的人工修改把 gather
排列换成它的逆排列，GEMM 消费端仍按原格式读数。两种排列各自都是双射；
错误出在**两个阶段组合以后不再还原同一个权重**。独立穷举完整 tile 的1024个
坐标，47 全部对应，48 有768个错位。

[T4 资格记录](private_data/qualification/r10.json) 使用直接 canonical 位解码、
H16 舍入和 FP64 求和，不调用优化 repack 或 consumer 索引作为数值参考。
两例各执行33个合法计划：47 的33项全过，最大误差/容差比为0.00007098；
48 有28项失败、5项通过，固定标签仍为 reject。首个反例 M3/K7/N7，输出
`0.21138522` 对参考 `-0.03153471`，误差0.24291993，允许0.01003153。
覆盖基向量、任意组排序、packing/tile 尾部、无效尾部位、signed words、非零
storage offset、变化的 scale/zero，以及 subnormal scale 压力；每次检查输入
按字节保持和输出独立存储。接受依据同时包括布局组合论证和全域保守误差界，
没有把33次测试当作全域证明。资格通过后冻结源码、合同、oracle 和标签。

### 人工 trace 审查与结论

**本批没有新增成功案例：三组均2/2。** 47–48 均归入 `ordinary_candidate`，
成功集合仍为 case_36、case_41。6次实验、32次SDK请求、213份trace文件均保留；
18次probe尝试对应14份脚本，不能称作18次有效实验。原始请求/响应、代码/合同
哈希、当前题隔离和verifier指纹核对通过，没有模型重跑或评测后的题目修改。
API估算合计$1.591777，不含GPU、非实际账单。

**无工具不是碰巧猜中。** 47用4,832输出token推导出了排序后组身份和
producer/consumer布局互逆。48用了20,041输出token、333.99秒，先多次错误
展开循环顺序并回忆不可靠的Marlin源码，最终回到题目实际代码，手算出word0的
8个nibble：消费者在`(k=1,n=0)`取lane4，生产者那里却存着`(k=8,n=0)`。
[原始推导](../traces_glm/case_48/single_call/r1/response_thinking.txt) 和
[最终判断](../traces_glm/case_48/single_call/r1/verdict.json) 与真实错配一致。
它没有完整指定activation/scale/zero并算出超差，不能记作可执行数值反例；
但也不能把这个正确的局部映射推导写成猜测或错误根因。

**solo使用GPU结果建立结论。** 48在工具前仅提出一般repack/group/zero风险，
没有定位inverse interleave；[t6](../traces_glm/case_48/solo/r1/probes/t6_stdout.txt)
36个合法对照全部失败后才确认格式路径有问题，随后
[t10](../traces_glm/case_48/solo/r1/probes/t10_stdout.txt) 又完成9个生成器样例和
1个M2/K16/N8微型反例，全部失败。t9先调用了候选一次，随后参考代码因设备
混用报错，因此共有46次有效数值对照、额外1次无有效对照的候选调用；t7被
流程拦截。最终正确reject，但未精确定位到inverse-interleave这一行，且
“几乎每个合法输入都错”是过度概括，零输入等可以通过。

47 solo的t7和t10共有162个合法对照通过。t6计划400种形状×4种数据，60.17秒
超时且stdout为空，实际完成次数未知；不能把计划1600次计作执行覆盖，也不能
写成零次调用。t8被流程拦截。已成功的测试未覆盖subnormal scale、storage
offset、输入保持和输出独立存储，比较也未显式检查finite；“full legal-domain
sweep”并不准确。正式登记的证据状态与两例solo最终判断一致。

**debate也有合法证据，但测试设计仍有缺口。** 47的t9/t10共有26个合法GPU
对照通过；t11的CPU数值检查实际检查0个scale、0种组合，却输出
`verdict_bit_identical=true`，不是成功的穷举。后续t16在CPU uint16 arange处
抛错，二者均没有调用候选GPU kernel。最终判断有静态数值理由，但不能宣称
H16/subnormal已经由这两段脚本验过。
Experimenter和Skeptic识别了空测试，保留c3为inconclusive；但对空测试原因的
说明有误：脚本把整数bit pattern数值转换成浮点数，没有reinterpret成FP16，
随后按scale范围过滤才全部为空，不能说已扫描subnormal数值。

48 debate共有24个合法GPU对照，22个失败、2个极小形状通过，足以固定reject。
不过[t16](../traces_glm/case_48/debate/r1/probes/t16_stdout.txt) 所谓的
`control_no_negative`仍有268个负qweight words、17个负qzero words；修改bit31
还改变了有效第8个nibble的数值。最大误差相同不代表排除了signedness影响，
不能把这次对照当作独立根因隔离。其他合法反例已经足以拒绝，不依赖这一点。
Skeptic和Judge没有识别这个伪对照，并沿用了“负words没有额外影响”的说法；
t16也没有正式finalize，c2仍为inconclusive。t9的finalize摘要还把实际3211个
违规输出写成3351；small-K失败只能说明这些输入上实现错误，不能据此确认
padding就是根因。最终reject证据足够，但debate没有给出精确interleave根因。

本轮确实提高了无工具的推导成本，工具提供了直接可复现的数值证据；但关键
错配仍可缩小为**单个word的8项位排列**，无工具在同样32,768输出token预算内
完成了推导。没有最终准确率增益，也没有证明debate优于solo，因此不扩为十例。
若继续研究，应先筛选必须跨多个阶段或运行语义才能确定的机制，而不能仅靠
加长流水线；仍须保留完整合同、固定实现标签与独立资格证据。


## 时间融合 Stencil：case_49–case_50（2026-10-03）

这是一对为工具比较构造的真实 stencil 家族合成实现，不声称复现上游历史 bug。
公开文件统一放在 `../triton_eval_cases/case_49` 和 `case_50`；构造、来源和实验记录集中在
`private_data/experiment_stencil.json`，独立验收程序为 `../eval_scripts/real_kernel_challenges/stencil_pair.py`。

合同：FP16 的二维输入，任意 H、W 从 1 到 129，绝对值不超过 1；执行 4 或 8 次
3×3 二项式平滑，每一步都重新施加零边界。目标是实数运算结果，允许 FP32 输出绝对误差 2e-5。
同一实现对整个公开域只有一个固定标签。输入只用于验证，不改变标签。

- **case_49 / accept**：四步融合成两次一维九点卷积，用奇对称周期扩展保持有限域的零边界。
- **case_50 / reject**：相同内部卷积系数，仅在每次融合时作一次零扩展，错误地允许离开有限域后再次进入的路径。

独立 CPU 验收对全部 129 种合法轴长度核对有限域算子四次方，共 723,905 个整数矩阵元素。
这证明所建模的线性算子等价，结合行列算子可交换覆盖二维数学公式。
每轴九点计算的绝对系数和不超过 1，最多四次轴扫描；保守 FP32 绝对误差界约 4.292e-6，低于预定容差。
另在 T4 核验实际编译、数值结果、形状/dtype、输入保持、输出独立存储、退化形状和 storage offset。
有限 GPU 测试本身不构成整个程序的形式化证明。

<!-- BEGIN REAL KERNEL STENCIL RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_49 | trust | [r1](../traces_glm/case_49/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0201 | [r1](../traces_glm/case_49/solo/r1/transcript.md)：trust / 正确；probe 2；$0.1054 | [r1](../traces_glm/case_49/debate/r1/transcript.md)：trust / 正确；probe 4；$0.3560 |
| case_50 | reject | [r1](../traces_glm/case_50/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0175 | [r1](../traces_glm/case_50/solo/r1/transcript.md)：reject / 正确；probe 1；$0.0914 | [r1](../traces_glm/case_50/debate/r1/transcript.md)：reject / 正确；probe 3；$0.2736 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 预算耗尽 | 运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.0375 |
| solo | 2 | 2 | 0 | 0 | 0 | 0 | 3 | $0.1968 |
| debate | 2 | 2 | 0 | 0 | 0 | 0 | 7 | $0.6296 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；运行失败 0，弃答 0，无最终答案 0，token 上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$0.8639**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（预算耗尽仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 2 / 2 | 100.0% |
| solo | 2 / 2 | 100.0% |
| debate | 2 / 2 | 100.0% |
原始提交批次：`run_20261003T221257_265100Z`、`run_20261003T221259_691345Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL STENCIL RESULTS -->


### 本轮 traces 人工复核

三组均为 2/2 正确，无工具约 $0.0375、solo 约 $0.1968、debate 约 $0.6296；
API 费用合计估算 $0.8639，不含 GPU。完整记录 6 次评测、24 次 SDK 请求、10 个 probe 脚本、172 个 trace 文件。
六次原始请求/响应、完整公开材料、预算、哈希、环境及隔离检查均通过。

| Arm | case_49 | case_50 |
|---|---|---|
| 无工具 | 静态推导出奇对称扩展维持零边界；数值界未完整推导 | 静态识别逐步截断与宽卷积不同；举的两步例子不在公开4/8步域，未给完整合法数值反例 |
| solo | 最初误判符号折叠有错，GPU证据纠正；11次kernel调用，10个数值比较通过 | 最初误认为融合正确，GPU证据纠正；70个已记录合法比较均失败，另有1次未单独报告的调用 |
| debate | 事先识别可能成立的边界恒等式；44次已知调用，41个可核查比较通过；1个probe空集合归约出错后修复 | 事先识别边界机制；8次已知调用，5个可核查比较失败；1个probe序列化出错后修复 |

工具确实纠正了两条 solo trace 中的错误初始假设，但无工具独立评测已经两题全对，
因此不能计作组间判断收益。两例均保留为 `ordinary_candidate`。

覆盖声明需收窄：49 solo 的7种shape并非全域穷举，输出独立存储也未实测；50 solo 的
输入保持检查比较同一存储、`.contiguous()` 消除了拟测试的偏移，而且“70/71”不能解释为一个通过。
49 debate 也没有完成全域代数或误差界证明；50 debate 的8步内部区域仍有误差，最后 verdict 才限定了4步内部近似一致的表述。
固定 accept 的全域数学依据来自独立验收，不能归给上述有限 probe。
逐脚本计数、限制及可复现反例详见 `private_data/experiment_stencil.json`。


## 整数重定标：case_51–case_52（2026-10-05）

这对合成 Triton 实现采用真实 gemmlowp 两级舍入语义，测试整数 GEMM 的后处理。
源码语义来源固定在 gemmlowp commit `16e8662c34917be0065110bfcd9cc27d30f52fdf` 的
`fixedpoint/fixedpoint.h`；没有声称发现或复现该项目的历史缺陷。
公开文件仍统一位于 `../triton_eval_cases/case_51` 和 `case_52`，来源、冻结信息与实验记录集中在
`private_data/experiment_requant.json`；独立验收程序为 `../eval_scripts/real_kernel_challenges/requant_pair.py`。

合同覆盖完整 int32 累加器值域、正 Q31 通道乘数、0–31 位右移、INT8 零点与饱和，
最后返回 FP32 反量化输出。明确要求两级舍入兼容性：第一层中点向正无穷，第二层中点远离零。
它不会把其他合同允许的单次舍入或量化近似判成错误。误差阈值固定为绝对 1e-6、相对 0。

- **case_51 / accept**：高低32位乘积融合两次舍入偏置，第二个低位进位基于第一次加法后的中间值。
- **case_52 / reject**：为并行计算进位，两个进位都根据原始低位乘积计算，遗漏加法之间的依赖。

Python 大整数参考直接执行两个舍入步骤，不复用 candidate 的高低位拆分。
正确性的全域依据来自整数进位恒等式、舍入合并恒等式及范围分析；GPU 测试另核验实际实现、边界、输入保持和别名规则。
所有模型看到同一完整合同；每个实现始终只有一个固定标签。保留所有三组实验，无区分度也保留。

模型调用前，T4 独立验收每例执行 22 个计划，合计 1,960,966 个输出。
51 全部精确一致；52 在 6 个计划失败，最大绝对误差 1。
例如 `a=-511,m=2^30,r=9,z=0,s=1`，实际输出 -1、参考 0。
storage offset、输入共享存储、输入保持及输出独立存储另有实际检查。
公开参考和私有 oracle 都直接实现同一合同公式，不能仅凭二者一致排除共同理解错误；
独立复核另用 gemmlowp 的 nudge/truncation 与 remainder/threshold 形式检查
204,032 个标量组合，全部一致。这是有限 Python 语义交叉检查，不是编译上游或完整程序形式证明。

<!-- BEGIN REAL KERNEL REQUANT RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_51 | trust | [r1](../traces_glm/case_51/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0714 | [r1](../traces_glm/case_51/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0988 | [r1](../traces_glm/case_51/debate/r1/transcript.md)：— / token 上限；probe 3；$0.3795 |
| case_52 | reject | [r1](../traces_glm/case_52/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0585 | [r1](../traces_glm/case_52/solo/r1/transcript.md)：trust / 错误；probe 3；$0.1785 | [r1](../traces_glm/case_52/debate/r1/transcript.md)：reject / 正确；probe 6；$1.1994 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.1299 |
| solo | 2 | 1 | 1 | 0 | 0 | 0 | 4 | $0.2773 |
| debate | 2 | 1 | 0 | 0 | 1 | 0 | 9 | $1.5788 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；按互斥结果类别计：其他运行失败 0，弃答 0，其他无最终答案 0，输出截断／上限 1，未完成 0。
全部尝试的已记录 API 费用估算：**$1.9860**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 2 / 2 | 100.0% |
| solo | 1 / 2 | 50.0% |
| debate | 1 / 2 | 50.0% |
原始提交批次：`run_20261005T065447_344341Z`、`run_20261005T065505_310931Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL REQUANT RESULTS -->

### 本轮 traces 人工复核

结论：**无工具 2/2 正确、solo 1/2 正确、debate 1/2 有效正确，另一次无最终判定**。
没有观察到工具相对无工具的判断增益；两例保留为 `ordinary_candidate`。
52 的 debate 胜过 solo 是本轮局部观察，但无工具同样正确，不能计入工具成功案例。
六次评测共 32 次 SDK 请求、13 次 probe 尝试（含一次未执行的 ledger 拒绝）、
12 个实际脚本、202 个 trace 文件。API 费用合计估算 $1.9860，不含 GPU。

| Arm | case_51 | case_52 |
|---|---|---|
| 无工具 | 纠正中间算术错误后给出分情况静态推导，正确接受；15,291 输出 tokens | 精确定位第二次进位使用旧 low，给出小值反例，正确拒绝；12,369 输出 tokens |
| solo | 742 次候选调用、950,784 个元素全部报告一致，正确接受；大量常数矩阵重复，不能称全域穷举 | 前两次 probe 超时／索引异常；最后仅 1 次合法调用、8,832 个元素通过，错误接受 |
| debate | 3 个脚本全因 shape 不合法而被拒绝，候选 GPU 调用为 0；后续响应截断导致解析失败，无 verdict | 4 次合法调用、637,728 个元素，6,371 条失败比较，正确拒绝；计数含跨 probe 重复，根因解释不准确 |

**52 solo 漏测的原因可以直接从代码确认。** 正负 R-stage tie 写入同一行，负数覆盖正数；
H-stage tie 找不到时静默填零；饱和边界搜索受 channel 0 的成功与否控制。
每个 shift 只绑定一个 multiplier、zero-point、scale；例如 r=1 绑定 z=127，可能遮住正数误差。
模型没有检查这些样本是否构造成功，却把“零误差”升级为整个合同成立。
400,000×32 的候选池只是 CPU 搜索材料，实际仅测试 276×32 个输出。
本机 PyTorch 2.9.1 的诊断重放得到 32 条 H-tie 行和 10 条饱和边界行全零，
以及 118 个中间整数错误被最终饱和掩盖；这是独立诊断重放，**不是原 T4/PyTorch 2.8.0 trace 的观测计数**。

**52 debate 的反例有效，解释却不成立。** 小值 sweep 先发现失败，再逐步排除部分舍入假设。
例如 t17 中 `a=-257,m=2^30,r=8,z=0,s=1`，实际 0、参考 -1，足以拒绝该实现。
但最终仍把问题归因于“融合两级舍入成为单次移位”。融合本身可以等价实现合同；
真正问题是第二次进位没有使用第一次加法后的低位值。
t17 的分 shift 计数还被后续 multiplier 覆盖，总失败数 1,029 正确，分组字典不能用作总覆盖。

**51 debate 是流程失败。** 两次 Skeptic 各耗 8,192 tokens 而未生成动作；随后三个脚本把
acc 写成 `[1,N]`，channel 参数却只有一个元素，全部在输入校验阶段失败。
Experimenter 最后识别出应改成 `[N,1]`，但请求仅剩 573 tokens，工具参数字符串被截断。
JSON 解析异常发生在切换收尾阶段之前：正常阶段实际已用 24,576，预留的 8,192 未使用。
这不是 kernel 错判或主动弃答，也不是用完了整个 32,768-token 配额。
汇总程序把它归为 `token_limit`，原始进程状态为 `error`，两种状态均保留。
全部 7 次 raw 请求、响应和 usage 保存完整；transcript/history 缺最后一次已计费调用，
费用按 raw responses 恢复为 $0.3794502，不能只累加 transcript。

这轮复杂度增加了无工具的推导成本，却没有阻止它得到正确答案；工具侧反而暴露出
样本生成没有验收、饱和掩盖错误、有限通过被扩大成全域保证，以及截断响应无法正常收尾的问题。
继续扩题前，应先离线修复截断参数到收尾的流程，并让测试报告实际构造成功的边界条件。
本轮未修改 verifier、提示、预算或重跑挑选结果；后续修复需要单独标记协议版本。
详细逐 probe 审计及原始调用用量保存在 `private_data/experiment_requant.json`。


## erf 全域数值精度：case_53–case_54（2026-10-09）

本轮测试一元误差函数 `erf` 的快速近似。五系数公式来自 Abramowitz–Stegun，
真实软件实例是 Microsoft.ML.CpuMath 的 `ProbabilityFunctions.cs`，固定于
dotnet/machinelearning commit `cd77dea510f09cec6e586ead43732877c1441bfe`。
这里是原创 Triton 移植和合成系数表修改，不声称复现该项目的历史 GPU 缺陷。
公开文件为 `../triton_eval_cases/case_53`、`case_54`；来源、构造搜索和三组记录
统一保存于 `private_data/experiment_erf.json`，独立验收程序为
`../eval_scripts/real_kernel_challenges/erf_pair.py`。

两例共享完整合同：一维连续 FP16 输入，长度 1–131072，全部有限 FP16 值合法，
包括正负零、subnormal 和 ±65504；允许 storage offset。输出为独立存储的有限 FP32，
保持输入。目标是数学定义的 erf，绝对误差不超过 **1e-6**，相对容差为零。
这个六位绝对精度要求在搜索候选之前确定，未根据两个实测误差挑选阈值。

- **case_53 / accept**：完整精度的五系数表。
- **case_54 / reject**：只将多项式系数表保存为五位小数，分母参数保持原精度。

CPU 初筛还检查了同时舍入分母参数及六位系数等方案，完整记录保留；有些看似降低精度的
方案仍符合合同，不能标为错误。两例已完成独立 T4 验收，标签在模型调用前冻结。
按 uint16 位模式重解释为 FP16，可覆盖 63,488 个有限编码；不能用整数数值转换冒充穷举。
MPFR 定向舍入构造真值区间，mpmath 高精度作独立交叉检查。数值域穷举覆盖实际执行的
标量路径；索引、尾块、输入保持和输出独立存储另行验收，不称整个程序的形式化证明。

三组保持 Fireworks GLM-5p3、相同公开材料、每组 32768 总输出上限与原有提示。
本轮先修复截断工具参数导致进程崩溃的问题：完整保存原始响应和 usage，记录无动作截断反馈，
允许现有预算收尾继续。新 verifier 指纹单独记录，旧 51–52 的失败记录不修改、不重跑。

<!-- BEGIN REAL KERNEL ERF RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_53 | trust | [r1](../traces_glm/case_53/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0027 | [r1](../traces_glm/case_53/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0734 | [r1](../traces_glm/case_53/debate/r1/transcript.md)：trust / 正确；probe 3；$0.2398 |
| case_54 | reject | [r1](../traces_glm/case_54/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0040 | [r1](../traces_glm/case_54/solo/r1/transcript.md)：reject / 正确；probe 1；$0.0764 | [r1](../traces_glm/case_54/debate/r1/transcript.md)：reject / 正确；probe 2；$0.2001 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 1 | 1 | 0 | 0 | 0 | 0 | $0.0067 |
| solo | 2 | 2 | 0 | 0 | 0 | 0 | 2 | $0.1498 |
| debate | 2 | 2 | 0 | 0 | 0 | 0 | 5 | $0.4399 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；按互斥结果类别计：其他运行失败 0，弃答 0，其他无最终答案 0，输出截断／上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$0.5963**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 1 / 2 | 50.0% |
| solo | 2 / 2 | 100.0% |
| debate | 2 / 2 | 100.0% |
原始提交批次：`run_20261009T070534_451596Z`、`run_20261009T070535_082295Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL ERF RESULTS -->

人工证据审计：

**这轮观察到 case_54 的工具流程增益；没有观察到 debate 超过 solo。**
三组只各运行一次，无工具 1/2，solo 2/2，debate 2/2。53 保留为正确对照及普通候选，
54 加入成功分组。两例都保留，合同、源码、容差和标签在模型调用前冻结，未重跑选最好结果。

独立 T4 验收每例运行 36 个计划、比较 991,333 个元素，包括四遍完整标量域、
不同排列、长度、尾块与 offset；参考使用 MPFR 320-bit 定向舍入区间，另以
mpmath 110 位交叉检查。53 的 36 个计划全部通过，最大真实误差上界为
`5.6488038806e-7`。54 有 29 个计划失败；每遍完整标量域中 1,394 个编码超差，
最大误差上界为 `1.2917548572e-6`。所有结构、有限性、输入及 backing guards 保持、
输出独立存储检查均通过，故本对差异来自数值精度。

| Case/组 | Trace 中实际执行的验证 | 审计结论 |
|---|---|---|
| 53 无工具 | 无执行；引用原 A&S 公式约 `1.5e-7` 误差界 | 判对，但引用的理想公式界没有覆盖实际 GPU 求值误差；未返回内部推理，不补写过程 |
| 54 无工具 | 读到五位系数、检查零点，仍套用原 A&S 误差界 | 错误接受；了解目标和 `1e-6` 阈值，问题是没有重新验证修改系数后的误差界 |
| 53 solo | 1 次 GPU，全部 63,488 个有限编码；对 FP32 舍入参考最大差 `5.6438148e-7` | 数值结论稳健；只运行一个整块长度，没有实测尾部、offset、输入保持和 fresh output，其“整个合同 100% 覆盖”表述过大 |
| 54 solo | 1 次 GPU，全部 63,488 个有限编码；对 FP32 舍入参考报 1,458 个超差 | 有真实反例；最初推理倾向通过，执行后改判拒绝，能看到工具证据纠正估计的过程 |
| 53 debate | 3 次 GPU：完整标量域、小值/subnormal、正负极值；均通过 | 增加了针对性检查，但仍不是整个张量合同的穷举；源码中的尾部 mask 被审阅 |
| 54 debate | 2 次 GPU：完整标量域及 FP64 多项式对照；找到相同最坏反例 | 拒绝正确，但枚举数量与误差归因都有瑕疵，见下文 |

**不能混用工具参考和数学真值的数字。** 两个 solo 都把 FP64 erf 舍入为 FP32 后比较。
54 的 trace 因此报告 1,458 个超差、最大差 `1.3113021851e-6`；独立 MPFR 验收对数学
目标给出 1,394 个、`1.2917548572e-6`。这不推翻拒绝：相同输入
`x=0.50830078125` 的实际输出为 `0.5277653932571411`，真值区间为
`[0.5277641015022839, 0.527764101502284]`，真实误差下界
`1.2917548571e-6` 已超过 `1e-6`。53 的数值裕量也足以容纳独立确认的参考舍入误差。

54 debate 将 int32 的所有位直接解释为 FP16，引入 65,536 个额外正零，故数组长度
129,024 **不是**不同有限 FP16 编码数。它仍包含全部 63,488 个唯一合法编码，N 也在
合同范围内；重复的零没有造成这次拒绝的错误，但“129,024 个有限值”的覆盖表述需纠正。

Debate 还把“FP32 求值偏差自身小于 `1e-6`”当成否定舍入致错的理由，这不成立。
已有独立 T4 记录中，`x=-0.400634765625` 的 FP64 多项式误差约 `9.79678e-7`，
本在容差内；实际求值额外偏差约 `2.42060e-7`，使总误差达到 `1.22173784e-6`。
因此正确解释是**系数修改与实际浮点求值共同决定最终误差**；即使另有输入的 FP64
多项式已经超差，也不能宣称舍入不参与失败。合法反例足以支持固定 reject，
并不使 trace 中每项分析都正确。

本次共 22 次 API 调用、7 次成功 GPU probe，158 个 trace 文件；全部 6 个槽位正常结束，
没有截断、丢答或重试。新截断恢复代码本轮未触发，不能将此次收益归因于该修复。
公开材料、哈希、隔离、预算与原始响应完整性检查无缺项，凭据扫描零命中。
运行时修复的 85 项离线回归、案例登记/Modal 收集的 70 项检查及报告的 6 项回归通过。
费用依保存的定价与 usage 估算为 **$0.596311**，GPU 未计入；不是账单。
该配对小实验支持继续研究可穷举数值域的近似实现，尚不能说明同类题上的稳定收益。


## 三轮顺序探索（2026-10-09）

每轮两个实现、三组实验，沿用 Fireworks GLM-5p3、low、32768 总输出预算、
工具单轮上限 8192 和现有隔离/收尾协议。不修改 verifier、提示或旧 traces。
读完本轮全部 traces，再据其证据选择下一轮；不预先生成三轮然后事后解释。
完整构造候选、资格记录、实际结果和每轮决策集中在
`private_data/experiment_adaptive_unary.json`，公开题目继续使用统一编号。

### 第一轮：GELU，case_55–case_56

上一轮的主要发现是：无工具把原公式的误差界直接套用到修改后的系数表，
solo 实测后纠正初始估计；debate 又漏掉了误差分项相加可能超界的问题。
因此本轮考察 erf 类近似经过 Gaussian tail 与乘法组合后的 GELU，
完整合同为全部有限 FP16 值、FP32 输出、数学目标 `x*Phi(x)`、固定绝对容差 `1e-6`。
容差在候选筛查前确定。数值域和张量属性分别验收；合法输入只是反例，不改变实现标签。

<!-- BEGIN REAL KERNEL GELU RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_55 | trust | [r1](../traces_glm/case_55/single_call/r1/transcript.md)：reject / 错误；probe 0；$0.0044 | [r1](../traces_glm/case_55/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0806 | [r1](../traces_glm/case_55/debate/r1/transcript.md)：trust / 正确；probe 2；$0.2129 |
| case_56 | reject | [r1](../traces_glm/case_56/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0780 | [r1](../traces_glm/case_56/solo/r1/transcript.md)：reject / 正确；probe 2；$0.1026 | [r1](../traces_glm/case_56/debate/r1/transcript.md)：reject / 正确；probe 4；$0.2814 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 1 | 1 | 0 | 0 | 0 | 0 | $0.0824 |
| solo | 2 | 2 | 0 | 0 | 0 | 0 | 3 | $0.1832 |
| debate | 2 | 2 | 0 | 0 | 0 | 0 | 6 | $0.4943 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；按互斥结果类别计：其他运行失败 0，弃答 0，其他无最终答案 0，输出截断／上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$0.7599**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 1 / 2 | 50.0% |
| solo | 2 / 2 | 100.0% |
| debate | 2 / 2 | 100.0% |
原始提交批次：`run_20261009T080012_772642Z`、`run_20261009T080012_812788Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL GELU RESULTS -->


第一轮人工审计与第二轮决策：

- 固定标签：55 accept（T4 最大误差上界 `5.267715415e-7`）；56 reject
  （`2.740334945e-6`，每遍全域 5,358 个编码超差）。两例各 36 个独立计划、
  991,333 次元素比较，结构和数学参考检查通过，0 个区间歧义。
- **55 新增工具流程增益**：无工具把原近似的粗全局误差界乘上最大输入后，当成
  已达到的反例，忽略尾部衰减；solo/debate 实测全部有限编码后接受。Debate 明确
  撤回了先前关于中大输入超差的怀疑。55 仍是一个固定 accept 实现，不随输入改标签。
- **56 不新增增益**：无工具最终以参数敏感度分析识别 `p=0.3276` 的影响，正常
  消耗 17,202 输出 tokens 后拒绝；并非因预算被迫猜答。两工具组也找到合法反例。
  Solo 把错误归为整个 A&S 方法的固有误差，未准确定位具体常数；debate 定位更好。
- 工具脚本仍有失败和错误声明：55 debate 首次含非法 NaN/Inf；56 solo 首次
  CPU/CUDA 比较失败；56 debate 一次 dtype 比较失败，修正后所谓输入保持检查
  是同一别名自比较，freshness 检查也只检查 dtype/device。合法数值反例仍成立，
  这些表述不能升级为已验证的张量属性。原始失败全部保留。
- Public reference 本轮直接返回 FP64，数值结果与独立区间一致；仍由私有 MPFR
  数学真值负责标签。较上一轮去掉了参考再次舍入 FP32 导致的错误计数歧义。
- 三组共 6 次评测，API 估算 **$0.759916**（不含 GPU），无模型重跑。

读完三组全部 traces 后，第二轮转向 **SiLU 的指数换底常数精度**，避免只重复
A&S 公式识别；继续测试输入相关的误差实际是否越界，并保留合格优化实现。
不修改模型协议，固定全有限 FP16、FP32 输出及绝对容差 `1e-6`，先验收再评测。

### 第二轮：SiLU，case_57–case_58

构造采用稳定尾项分解：`a=abs(x)`，`u=exp2(-a*LOG2_E)`，
`tail=a*u/(1+u)`；正输入输出 `x-tail`，负输入输出 `-tail`。
数学目标仍为 `x/(1+exp(-x))`，实现允许近似但必须在完整合同下符合固定精度。
这里只改变换底常数的保存精度；候选筛查、来源及后续结果保存在同一三轮 manifest。

<!-- BEGIN REAL KERNEL SILU RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_57 | trust | [r1](../traces_glm/case_57/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0041 | [r1](../traces_glm/case_57/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0751 | [r1](../traces_glm/case_57/debate/r1/transcript.md)：trust / 正确；probe 1；$0.1730 |
| case_58 | reject | [r1](../traces_glm/case_58/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0161 | [r1](../traces_glm/case_58/solo/r1/transcript.md)：reject / 正确；probe 2；$0.1009 | [r1](../traces_glm/case_58/debate/r1/transcript.md)：reject / 正确；probe 3；$0.2273 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.0202 |
| solo | 2 | 2 | 0 | 0 | 0 | 0 | 3 | $0.1761 |
| debate | 2 | 2 | 0 | 0 | 0 | 0 | 4 | $0.4003 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；按互斥结果类别计：其他运行失败 0，弃答 0，其他无最终答案 0，输出截断／上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$0.5966**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 2 / 2 | 100.0% |
| solo | 2 / 2 | 100.0% |
| debate | 2 / 2 | 100.0% |
原始提交批次：`run_20261009T081207_003512Z`、`run_20261009T081208_394899Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL SILU RESULTS -->


第二轮人工审计与第三轮决策：

- **三组都是 2/2，本轮没有新增分类收益。** 57 固定 accept，全域最大误差上界
  `9.490936108e-7`，容差余量约 `5.09e-8`；58 固定 reject，全域 3,427 个超差编码，
  最大误差上界 `1.651000610e-6`。两例全部结构检查通过，36 个独立计划各比较 991,333 元素。
- 无工具在 57 正确识别稳定尾部结构，但“精度余量充足”的解释没有量出实际最坏位置。
  58 用 3,184 输出 tokens 推导换底常数扰动的敏感度，在 `|x|≈2.4` 估计出超差区域，
  说明换一个激活函数并没有让明显常数修改变得必须依靠工具。
- Solo 的 58 首次脚本将 int32 按位解释为 int16，插入 65,536 个额外零；仍包含
  全部 63,488 个有限编码且 N 合法。它已完成比较、打印 3,427 个超差后，才因
  CPU/CUDA 索引打印失败；不能将这次脚本算作没有有效数值证据。随后正确枚举复现。
- Debate 两例均正确，57 全标量域 1 次调用，58 做定向扫描、真正全域扫描和
  offset 扫描共 3 次。它的输入保持检查仍是运行后别名自比较，不构成保持性证明。
  正确拒绝由真实数值反例支持，不依赖该错误属性检查。
- 原始记录全部保留；6 次评测 API 估算 **$0.596577**（不含 GPU），没有模型重跑。

读完本轮三组 traces 后，第三轮转向 **正弦的参数缩减与浮点求值方式**：
同样常数与数学公式，比较 fused multiply-add 和分别舍入的乘减，减少常数截短线索。
固定数学 sin、全有限 FP16 输入、FP32 输出、绝对容差 `1e-6`，不根据模型结果调阈值。
私有验收除数学参考与 GPU 实测外，另核对编译后的 PTX 是否采用预期算术。

### 第三轮：正弦参数缩减，case_59–case_60

通过分拆的 π/2 常数将输入缩减到短多项式区间，再按象限恢复 sin。
两例使用相同常数与多项式，只改低位修正的一行：59 使用显式 `tl.fma`，
60 使用分别舍入的乘法和减法；两者关闭自动融合。固定容差 `1e-6` 后只筛查
这两个候选，不按模型结果调阈值。机制参考 Cephes，具体 π/2 分拆、Taylor
多项式与 Triton 实现是人工适配，不声称复现上游历史 bug。

独立 MPFR 区间与 T4 验收 `qualification/r16.json`：59 全部 36 个计划通过，
全标量域最大误差上界 `1.339440121e-7`；60 固定 reject，25/36 个计划通过，
四次全域扫描均只有 `±63168` 两个编码超差，最大误差上界 `1.003946636e-6`。
反例的误差下界也严格超过容差，超出约 `3.9466e-9`，没有 oracle 区间歧义。
这说明故障稀疏，不说明 kernel 随输入改变标签。

每例主验收比较 991,333 个元素，另有 N=3 的一次 PTX 检查及 3 个数值比较。
PTX 中 59 有显式 FMA，60 为分开的乘减；PTX 结论只针对该特化。实际最大
缩减余量约 `0.787903`，并非严格小于 π/4；独立 oracle 不复用候选的缩减或多项式。
全标量枚举、张量属性检查与源码索引分析分别记录，不称全张量域形式证明。

<!-- BEGIN REAL KERNEL SINE RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_59 | trust | [r1](../traces_glm/case_59/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0305 | [r1](../traces_glm/case_59/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0744 | [r1](../traces_glm/case_59/debate/r1/transcript.md)：trust / 正确；probe 2；$0.2052 |
| case_60 | reject | [r1](../traces_glm/case_60/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0092 | [r1](../traces_glm/case_60/solo/r1/transcript.md)：reject / 正确；probe 1；$0.0784 | [r1](../traces_glm/case_60/debate/r1/transcript.md)：reject / 正确；probe 5；$0.7028 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.0397 |
| solo | 2 | 2 | 0 | 0 | 0 | 0 | 2 | $0.1529 |
| debate | 2 | 2 | 0 | 0 | 0 | 0 | 7 | $0.9080 |

目标 **6 个 case×arm 槽位**：已有记录 6，缺失 0。共有 6 次尝试；按互斥结果类别计：其他运行失败 0，弃答 0，其他无最终答案 0，输出截断／上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$1.1006**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **6 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 2 / 2 | 100.0% |
| solo | 2 / 2 | 100.0% |
| debate | 2 / 2 | 100.0% |
原始提交批次：`run_20261009T082756_036807Z`、`run_20261009T082758_028490Z`。按 rN 核验共同 verifier 指纹；保留原始提交身份。
已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。
样本量仅 2 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL SINE RESULTS -->

第三轮人工审计：

- **三组都是 2/2，没有新增分类收益。** 无工具对 59 用 6,412 输出 tokens
  自行纠正早期的消去误差猜测，识别 `1.5703125=201/128` 的精确高位乘积和
  FMA 单次舍入，给出有实质内容但尚不严密的误差估计。不能说它只靠猜测接受。
- 无工具对 60 判对但原因错：没有约简 `804/512`，错误声称高位乘积需要
  25 位、引入约 `2e-3` 误差。实际 `|q|<=41701`，`|q*201|<2^23`，
  高位乘积在 FP32 中精确。真实最坏误差只有 `1.003946636e-6`，仅两个编码超差。
- Solo 两例各一次 GPU probe，均比较全部 63,488 个有限 FP16 编码，复现私有验收。
  60 有合法反例，但最终解释仍误将高位乘积列作误差来源。59 只有一个 N、零偏移，
  没有执行输入保持/输出独立性检查；“没有其他可能失败模式”超出实际证据。
- Debate 的 59 两次都只扫描 `|x|>=1024`：共 24,576 个比较、12,288 个不同编码，
  未测其余 51,200 个编码。最终把遗漏区间概括为 `q=0`、计算精确不成立
  （例如 x=1 的 q=1）。判断正确，不代表接受依据完整。
- Debate 的 60 有 5 次 probe 请求，其中 t15 因前一证据尚未归档而被 ledger
  拒绝，未运行脚本；其余 4 次 GPU 调用比较 24,128 元素。普通随机数和极值
  均未揭示错误，定向扫描发现 -63168，随后大值域枚举复现 ±63168。
  额外讨论纠正了记录状态并复现反例，但没有纠正最终的“高位相减消去误差”归因。
- 反例的直接算术解释：x=63168 时 q=40214，高位乘积精确为 63148.546875，
  r_hi=19.453125；FP32 低位乘积先舍入产生约 `-9.00938e-7` 的误差。
  分步计算的残差为 `-0.003484725952`，FMA 为 `-0.003485627007`。
  这份 CPU 诊断只用于解释；标签仍以先前冻结的 MPFR/T4 验收为依据。
- 模型两组工具流程都没有检查 PTX；编译结果检查是作者侧独立验收，不能算作
  模型自行取得的工具收益。本轮 API 估算 **$1.100639**，不含 GPU，没有模型重跑。

### 三轮汇总与后续判断

每轮完整读完三组 traces 后才决定下一轮方向；每轮构造、CPU 候选、独立验收、
源码冻结、预检和人工审计集中在
[同一实验记录](private_data/experiment_adaptive_unary.json)。每例是完整合同下的
一个固定标签，不根据数据、模型回答或测试通过率改变。三个接受版本与三个拒绝
版本全部保留，不只统计胜例；这是主动构造的小样本探索，不是独立随机测试集。

| 轮次 | Case / 方向 | 无工具正确 | Solo 正确 | Debate 正确 | API 估算 |
|---|---|---:|---:|---:|---:|
| 1 | 55–56 / GELU 近似误差 | 1/2 | 2/2 | 2/2 | $0.759916 |
| 2 | 57–58 / SiLU 换底常数精度 | 2/2 | 2/2 | 2/2 | $0.596577 |
| 3 | 59–60 / 正弦缩减与 FMA | 2/2 | 2/2 | 2/2 | $1.100639 |
| 合计 | 6 个不同实现 | **5/6** | **6/6** | **6/6** | **$2.457132** |

按 arm 分别为无工具 $0.142317、Solo $0.512090、Debate $1.802725。
以上费用据原始 API usage 与定价记录估算，不是账单，不含 GPU。18 个评测槽位
全部完成，77 次 SDK 调用的请求/响应/usage 完整；无弃答、输出截断或模型重跑。
25 次 probe 请求包含被拦截及执行失败的尝试，不能等同于 25 次有效 GPU 验证。
协议、公开材料和 verifier 指纹核验通过，76 项相关离线测试通过。

新增工具流程胜例只有 **55**：正确优化实现被无工具的过松误差估计误拒绝，
两组工具验证得到正确接受。56–60 登记为 `ordinary_candidate`；全部原始 traces
仍在统一路径。这里的胜例是本次单次观察，尚未复测证明稳定优势。

下一步设计的依据是：

1. **保留能通过合同的优化实现。** 本次收益来自纠正过度拒绝；仅制造显眼的
   精度降级，容易被静态推导识别，也容易让“总是拒绝”获得虚假的好成绩。
2. **公式更复杂或故障更稀疏，本身不保证区分。** 59 的 FMA 可被推导理解，
   60 虽只有两个失败编码，无工具仍以错误理由判对。不能把解释改善算成分类增益。
3. **工具应支持检查假设。** 除实际输出，还可查中间运算的精确性、编译指令、
   索引覆盖和存储属性。这次多组最终解释与覆盖声明有误，增加角色没有自动解决。
   后续可探索误差抵消/衰减与多阶段求值，使粗上界不足以判断，同时保留固定合同、
   成对接受对照和独立参考；必须继续实测，不能预设会胜过无工具。

本轮没有观察到 Debate 超过 Solo 的准确率收益。

## Cases 61–66: three new directions

2026-10-17, America/New_York. Six new synthetic implementations were independently qualified on NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0 before model evaluation. Each direction contains one Accept implementation and one Reject implementation under the same public requirements. Labels apply to the whole legal domain; a failing input is evidence against the implementation, not a separately labelled example. Sources, tolerances, candidate searches and qualification hashes are recorded in the three existing family manifests linked below. No implementation, tolerance or label was changed after model evaluation.

All 18 evaluations used Fireworks `accounts/fireworks/models/glm-5p3`, reasoning effort `low`, and the same verifier fingerprint. Each arm had a total output limit of 32,768 tokens. Tool turns were limited to 8,192 tokens; Solo allowed 10 rounds and Debate 4 rounds with the existing closeout reserve. Every arm received the same complete public problem and kernel for its case. Private answers, qualification data, other cases and previous traces were excluded. All evaluations completed on their first attempt; probe-script failures and their repairs remain in the traces.

### 61–62: reciprocal square root

The target is `x / sqrt(1 + x*x)`, with FP32 output and absolute error at most `1e-6` for every finite FP16 input. The implementation uses an integer-bit reciprocal-square-root seed followed by Newton refinement. Case 61 uses three refinements and passes; case 62 uses two and fails. The algorithm is inspired by the [classic reciprocal-square-root implementation](https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/q_math.c); these are original synthetic Triton kernels, not claimed upstream defects. [Construction and complete results](private_data/source_hypot_pair.json).

<!-- BEGIN REAL KERNEL HYPOT RESULTS -->

| Case | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|
| case_61 | Accept | [Accept](../traces_glm/case_61/single_call/r1/transcript.md) (correct) | [Accept](../traces_glm/case_61/solo/r1/transcript.md) (correct) | [Accept](../traces_glm/case_61/debate/r1/transcript.md) (correct) |
| case_62 | Reject | [Accept](../traces_glm/case_62/single_call/r1/transcript.md) (wrong) | [Reject](../traces_glm/case_62/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_62/debate/r1/transcript.md) (correct) |

<!-- END REAL KERNEL HYPOT RESULTS -->

Case 62 is the new observed tool-benefit case. The no-tool model first recalled a two-step error near `4e-6`, then replaced it with an incorrect recurrence, `e_next ≈ e²/2`, and accepted. For reciprocal-square-root Newton iteration in exact arithmetic, the relative-error recurrence is `e_next = -1.5e² - 0.5e³`. Its mistaken coefficient substantially reduced its estimate.

Excerpt from the actual [case 62 kernel](../triton_eval_cases/case_62/kernel.py):

```python
for _ in tl.static_range(2):
    inverse_length = inverse_length * (
        1.5 - half_squared_length * (inverse_length * inverse_length))
value = x * inverse_length
```

Solo also initially estimated an error around `1e-7`. It then correctly reinterpreted all FP16 bit patterns, filtered NaN and infinity, ran the actual kernel on all 63,488 finite encodings, and found 20,050 violations. Debate independently found the same failure. A legal witness is `x=1640`: output `0.9999951124191284`, FP64 reference `0.9999998140988026`, error `4.701679674e-6`, exceeding the allowed `1e-6`. The [Solo probe and output](../traces_glm/case_62/solo/r1/probes/t6_stdout.txt) support rejection without relying on the model’s error estimate. Case 61 provides the passing control: independent maximum error is bounded by `1.680985035e-7`.

### 63–64: Softplus

The target is the [Softplus activation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html), `log(1 + exp(x))`, evaluated against a stable mathematical reference. These kernels approximate the logarithm after range reduction. Case 63 keeps terms through `q⁷`; case 64 stops at `q⁵`. Both use absolute tolerance `1e-6`. [Construction and complete results](private_data/experiment_softplus.json).

<!-- BEGIN REAL KERNEL SOFTPLUS RESULTS -->

| Case | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|
| case_63 | Accept | [Accept](../traces_glm/case_63/single_call/r1/transcript.md) (correct) | [Accept](../traces_glm/case_63/solo/r1/transcript.md) (correct) | [Accept](../traces_glm/case_63/debate/r1/transcript.md) (correct) |
| case_64 | Reject | [Reject](../traces_glm/case_64/single_call/r1/transcript.md) (correct) | [Reject](../traces_glm/case_64/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_64/debate/r1/transcript.md) (correct) |

<!-- END REAL KERNEL SOFTPLUS RESULTS -->

This direction did not separate the arms. The no-tool model correctly estimated the omitted term: at `x=0`, case 64 returns `0.6931458711624146` instead of `log(2)=0.6931471805599453`, an error of about `1.3094e-6`. The tool runs confirmed the defect. Case 63 passed the complete scalar sweep with a maximum error bound of `5.778695354e-7`. These remain useful correct/incorrect controls.

### 65–66: cancellation in a loss function

The target is the pseudo-Huber loss `sqrt(1 + x*x) - 1`, a robust loss related by a constant scale to [Ceres SoftLOneLoss](https://ceres-solver.googlesource.com/ceres-solver/+/master/include/ceres/loss_function.h). Case 65 uses the equivalent expression `x*x / (sqrt(1 + x*x) + 1)`; case 66 subtracts directly. The fixed requirement is `abs(error) <= 1e-8 + 1e-5*abs(target)`. For small inputs, the direct subtraction can lose significant digits. [Construction and complete results](private_data/experiment_compensated.json).

<!-- BEGIN REAL KERNEL COMPENSATED RESULTS -->

| Case | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|
| case_65 | Accept | [Accept](../traces_glm/case_65/single_call/r1/transcript.md) (correct) | [Accept](../traces_glm/case_65/solo/r1/transcript.md) (correct) | [Accept](../traces_glm/case_65/debate/r1/transcript.md) (correct) |
| case_66 | Reject | [Reject](../traces_glm/case_66/single_call/r1/transcript.md) (correct) | [Reject](../traces_glm/case_66/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_66/debate/r1/transcript.md) (correct) |

<!-- END REAL KERNEL COMPENSATED RESULTS -->

This direction also did not separate the arms. The no-tool model recognized both the stable rewrite and the cancellation defect. Independent validation found zero failing scalar encodings for case 65 and 15,728 for case 66. The worst error-to-tolerance ratios were approximately `0.01532` and `9.510`, respectively; a ratio above one fails. The ratio matters here because the allowed absolute error varies with the mathematical target.

### Validation and evidence limits

Independent qualification used directed high-precision MPFR bounds, a separate mpmath cross-check, and actual GPU execution. Each implementation received 36 validation plans, including all 63,488 finite FP16 encodings in multiple arrangements, boundary lengths, storage offsets, input preservation and separate output storage. All three Accept implementations passed every plan. The qualification records are [61–62](private_data/qualification/r17.json), [63–64](private_data/qualification/r18.json) and [65–66](private_data/qualification/r19.json). Scalar enumeration is exhaustive for values, not a formal proof of every tensor execution or GPU environment.

Trace audit found two recurring limitations. First, several accepting agents claimed the entire legal input domain was verified after a scalar sweep, despite incomplete checks of lengths, offsets and storage behavior. Second, case 62 Debate attached an unsupported explanation about intermediate rounding to its valid counterexample: `65504²` is exactly representable in FP32, and the extrema probe did not isolate the claimed cause. The primary counterexample still establishes rejection. Script errors were retained rather than hidden: these include invalid finite-value assertions, CPU/CUDA comparison mismatches and a malformed dtype constructor.

Case 65 Debate used targeted value and offset tests rather than a complete scalar sweep. Case 66 Debate found valid failures, but confused a particular counterexample’s ratio (`6.4225`) with the global maximum ratio (`9.5099`) at another input. Its three successful bit-pattern bands together cover the finite scalar domain, with overlapping samples; their failure counts must not be added as if disjoint. These details are preserved in the family manifest.

### Totals

| Arm | Correct | Wrong | Unfinished | API estimate |
|---|---:|---:|---:|---:|
| single_call | 5/6 | 1 | 0 | $0.040780 |
| solo | 6/6 | 0 | 0 | $0.485437 |
| debate | 6/6 | 0 | 0 | $1.665735 |

Total API estimate: **$2.191952**, across 18 evaluations and 71 recorded model API calls. This is a conservative estimate from captured usage and pricing, not an invoice; Modal GPU costs are excluded. All request/response captures, source hashes, model settings, budgets and the shared runtime fingerprint passed the structural audit. Probe evidence was reviewed separately.

Keep case 62 with the observed tool-benefit cases. Keep 61 and 63–66 as controls or ordinary-sample candidates. This wave adds one observed tool benefit in six cases and provides no accuracy advantage for Debate over Solo. The initial wave has one run per case and arm. The subsequent case 62 replication is reported below. The most promising follow-up is whether a realistic approximation reaches a stated tolerance after a specific number of refinements; simply adding more code or a familiar cancellation pattern did not produce a gap here.

### Case 62 replication

A fresh r2 repeated all three arms with the same source, public requirements, model, budgets and runtime. The kernel defect reproduced, but the tool/no-tool separation did not.

| Trial | No tools | Solo | Debate |
|---|---|---|---|
| r1 | [Accept (wrong)](../traces_glm/case_62/single_call/r1/transcript.md) | [Reject (correct)](../traces_glm/case_62/solo/r1/transcript.md) | [Reject (correct)](../traces_glm/case_62/debate/r1/transcript.md) |
| r2 | [Reject (correct)](../traces_glm/case_62/single_call/r2/transcript.md) | [Reject (correct)](../traces_glm/case_62/solo/r2/transcript.md) | [Reject (correct)](../traces_glm/case_62/debate/r2/transcript.md) |

No-tool r2 recognized the residual-error scale after two refinements and considered inputs near `sqrt(3)`. Its recorded output increased from 558 to 7,405 tokens under the same 32,768-token limit. Its arithmetic remains approximate; the proposed input was not executed. Both tool arms again correctly enumerated all 63,488 finite FP16 encodings and found 20,050 violations, maximum error `4.701679674e-6` at `x=±1640`.

The single-call requests are identical between trials. Tool initial requests differ only in bootstrap event timestamps. Source hashes, runner/verifier fingerprints, runtime and settings match. Both tool probes have valid evidence: [Solo stdout](../traces_glm/case_62/solo/r2/probes/t6_stdout.txt) and [Debate stdout](../traces_glm/case_62/debate/r2/probes/t8_stdout.txt). Full-sweep statistics must be read from stdout: Solo prints Python dictionary text, and Debate prints two JSON objects while its parsed result retains only the final extrema object.

The three reruns cost an estimated **$0.342486** in model API usage, excluding GPU charges. All 12 raw API calls were captured and checked. Across two trials, no tools is correct in 1/2 trials, Solo in 2/2 and Debate in 2/2. Keep the original observed gain, mark the separation **unstable**, and do not present case 62 as a reliable tool-only discriminator.

## Integer arithmetic and ordering: cases 67–72

2026-10-17, America/New_York. This wave tests three directions beyond approximation error: exact integer division, transform coefficient ordering, and stable argmax ties. Each direction has two different implementations under the same public requirements, one Accept and one Reject. They are synthetic adaptations of real algorithmic techniques, not claimed historical bugs. Every label was frozen after independent CPU analysis and NVIDIA T4 qualification. A legal failing input establishes a defect in the implementation; it does not receive a separate label.

All six cases received one evaluation from each arm using Fireworks `accounts/fireworks/models/glm-5p3`, reasoning effort `low`, and the existing 32,768-token total output budget. Tool turns used 8,192 tokens, Solo 10 rounds, and Debate 4 rounds with the existing closeout reserve. The source, public requirements and verifier were unchanged between arms. Before any model call, the plan required one additional three-arm trial for each initial no-tool error with valid evidence from a correct tool arm. Only case 68 triggered that rule. The selected replication is reported separately from the full 18-evaluation wave.

### 67–68: integer division for indexing

The kernel converts a flat unsigned index into a row and column for a width-seven layout: `row = index // 7`, `column = index % 7`. Every `uint32` value is legal, and the answer must be exact. Reciprocal multiplication is used in GPU index calculation, including [CUTLASS FastDivmod](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/fast_math.h). These implementations adapt that technique. Case 67 uses a correct round-down reciprocal with a widened increment. Case 68 uses a round-up reciprocal that is insufficient for the full unsigned input range. [Construction, proof and trace audit](private_data/source_fastdiv_pair.json).

<!-- BEGIN REAL KERNEL FASTDIV RESULTS -->

| Case | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|
| case_67 | Accept | [Accept](../traces_glm/case_67/single_call/r1/transcript.md) (correct) | [Accept](../traces_glm/case_67/solo/r1/transcript.md) (correct) | [Accept](../traces_glm/case_67/debate/r1/transcript.md) (correct) |
| case_68 | Reject | [Accept](../traces_glm/case_68/single_call/r1/transcript.md) (wrong) | [Reject](../traces_glm/case_68/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r1/transcript.md) (correct) |

<!-- END REAL KERNEL FASTDIV RESULTS -->

The central calculation in [case 68](../triton_eval_cases/case_68/kernel.py) is:

```python
# value is uint64; the public input covers the full uint32 range.
quotient = (value * 2454267027) >> 34
remainder = value - quotient * 7
# Both outputs are then stored as uint32.
```

For the legal input `3435973841`, the correct result is `[490853405, 6]`. The kernel returns `[490853406, 4294967295]`: the quotient is one too large, and the negative remainder wraps when stored as unsigned. Independent analysis divides the full domain into seven remainder classes and identifies exactly 122,713,351 failing inputs. Solo r1 also enumerated the CPU integer formula over all `2^32` inputs, then reproduced the defect with the actual GPU kernel. Debate r1 found exact GPU mismatches near the upper end of the domain. Debate r2 ran all 122,713,351 predicted failing inputs on the GPU in legal-size batches and confirmed every failure. This last sweep covers the predicted failing set, not every `uint32` value.

No-tool r1 accepted after both miscalculating the multiplier and misapplying its validity condition. In r2 it calculated the multiplier correctly, but argued that a reciprocal error below `0.5` guarantees the same integer quotient. That argument is false: when the exact division has remainder six, the fractional part is `6/7`, only `1/7` below the next integer. The formula can cross that boundary. The tool runs provide executed counterexamples instead of relying on this estimate.

| Case 68 trial | No tools | Solo | Debate |
|---|---|---|---|
| r1 | [Accept](../traces_glm/case_68/single_call/r1/transcript.md) (wrong) | [Reject](../traces_glm/case_68/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r1/transcript.md) (correct) |
| r2 | [Accept](../traces_glm/case_68/single_call/r2/transcript.md) (wrong) | [Reject](../traces_glm/case_68/solo/r2/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r2/transcript.md) (correct) |

In the initial r1/r2 trials, no tools accepted twice, while Solo and Debate rejected twice with valid GPU counterexamples. The subsequent r3–r5 check below found that this separation is intermittent. Initial single-call requests are identical; tool initial requests differ only in bootstrap timestamps. Two trials are not enough to establish a stable success rate. The tool agents sometimes suspected the defect before execution, so the evidence supports a workflow benefit without isolating tool access from additional reasoning. Keep case 68 with the observed tool-benefit cases. It does not establish an accuracy advantage for Debate over Solo.

### 69–70: transform coefficient order

These kernels compute the integer Walsh–Hadamard transform and return its coefficients in sequency order, ordered by the number of sign changes in each basis row. The arithmetic is exact over the full `int16` value range for widths 32, 64 and 128. Case 69 applies `bitreverse(gray(k))`; case 70 reverses those operations. The additions are correct in both, but case 70 places coefficients in the wrong positions. This is an original Triton implementation of a [standard transform](https://www.mathworks.com/help/signal/ref/fwht.html). [Construction and trace audit](private_data/source_butterfly_pair.json).

<!-- BEGIN REAL KERNEL BUTTERFLY RESULTS -->

| Case | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|
| case_69 | Accept | [Accept](../traces_glm/case_69/single_call/r1/transcript.md) (correct) | [Accept](../traces_glm/case_69/solo/r1/transcript.md) (correct) | [Accept](../traces_glm/case_69/debate/r1/transcript.md) (correct) |
| case_70 | Reject | [Reject](../traces_glm/case_70/single_call/r1/transcript.md) (correct) | [Reject](../traces_glm/case_70/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_70/debate/r1/transcript.md) (correct) |

<!-- END REAL KERNEL BUTTERFLY RESULTS -->

This direction did not produce a no-tool classification error. For case 70, the no-tool model initially thought the ordering was standard, then spent 6,884 output tokens working through it and correctly rejected. Its small examples used widths outside the public domain, so they are explanatory calculations, not legal executed counterexamples. Independent qualification provides a legal witness at width 32: input `[0, 1, 0, ...]` gives coefficient 8 equal to `-1`, where the reference requires `+1`.

Case 69 Debate used a useful verification method: it evaluated every basis vector at each legal width and compared all 21,504 matrix coefficients exactly. Combined with linearity and the proven integer bound, this checks the transform itself instead of relying only on random vectors. It also checked nonzero storage offsets. The no-tool model nevertheless reached the correct label. The stronger evidence did not improve accuracy on this pair.

### 71–72: the first index among equal maxima

These kernels return each row's first maximum index. They pack an FP16 value rank and an inverted column index into a single integer key, then reduce with `max`. The numerical values `-0.0` and `+0.0` are equal, so `[-0.0, +0.0]` must select index zero. Case 71 gives both zeros the same rank; case 72 does not and incorrectly selects index one. The design adapts float-bit ordering and packed reductions; it is not an upstream bug claim. [Construction and trace audit](private_data/source_packed_argmax_pair.json).

<!-- BEGIN REAL KERNEL PACKED ARGMAX RESULTS -->

| Case | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|
| case_71 | Accept | [Accept](../traces_glm/case_71/single_call/r1/transcript.md) (correct) | [Accept](../traces_glm/case_71/solo/r1/transcript.md) (correct) | [No verdict](../traces_glm/case_71/debate/r1/transcript.md) (error) |
| case_72 | Reject | [Reject](../traces_glm/case_72/single_call/r1/transcript.md) (correct) | [Reject](../traces_glm/case_72/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_72/debate/r1/transcript.md) (correct) |

<!-- END REAL KERNEL PACKED ARGMAX RESULTS -->

The no-tool model recognized both the correct zero handling and the defective version directly from source. This pair is useful as a control, but did not separate no tools from the tool workflows. Case 71 Debate produced malformed tool-call JSON before executing a probe and ended without a verdict. Its response, error and cost are retained; it was not rerun to improve the result.

### Evidence and totals

Independent [T4 qualification](private_data/qualification/r20.json) used PyTorch 2.8.0 and Triton 3.4.0. It combined full-domain integer analysis for division, complete operator checks for the transform, and all 63,488 finite FP16 encodings for value ordering, with GPU shape, offset, input-preservation and separate-output tests. Cases 67, 69 and 71 passed all 33, 66 and 111 validation plans respectively. These arguments and checks support the fixed labels; they are not a formal proof of every possible GPU execution.

Trace review separates correct verdicts from the quality of their explanations. Case 67 Solo repaired an oversized input and a timed-out CPU sweep before completing full scalar enumeration; case 68 Solo r2 repaired unsupported unsigned PyTorch operations before finding valid counterexamples. Some summaries overstate coverage or misreport values: 71 Solo sampled 32,768 bit patterns rather than all 63,488 finite encodings; 67 Debate reported the wrong remainder for the maximum input despite correct comparisons in its probe; 70 Solo called an output magnitude an error magnitude. Full findings and probe paths are retained in the family manifests. The 71 Debate failure was malformed JSON with finish reason `tool_calls`, not a token-limit response. All raw responses, including that failure, passed the capture audit.

| First-pass arm | Correct | Wrong | Unfinished | API estimate |
|---|---:|---:|---:|---:|
| single_call | 5/6 | 1 | 0 | $0.058099 |
| solo | 6/6 | 0 | 0 | $0.595463 |
| debate | 5/6 | 0 | 1 | $1.216206 |

The full first pass cost an estimated **$1.869768**. The selected case 68 replication cost **$0.434908**, giving **$2.304676** across 21 evaluations. These are conservative model API estimates from captured usage, excluding Modal GPU charges. Keep 67 and 69–72 as ordinary-sample candidates; retain the failed Debate run in the totals. The follow-up direction is exact integer optimization across width and range boundaries. Familiar signed-zero mistakes were easy to catch from source, while longer transform reasoning still produced a correct no-tool answer.

### Case 68 stability check

Three additional trials (r3–r5) were declared before any new API call. All nine evaluations completed under the same model, public source, problem, budget and verifier as r1/r2. No trial was dropped or rerun to improve its answer. The fixed kernel label remains **Reject**.

| Trial | No tools | Solo | Debate |
|---|---|---|---|
| r1 | [Accept](../traces_glm/case_68/single_call/r1/transcript.md) (wrong) | [Reject](../traces_glm/case_68/solo/r1/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r1/transcript.md) (correct) |
| r2 | [Accept](../traces_glm/case_68/single_call/r2/transcript.md) (wrong) | [Reject](../traces_glm/case_68/solo/r2/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r2/transcript.md) (correct) |
| r3 (new) | [Reject](../traces_glm/case_68/single_call/r3/transcript.md) (correct) | [Reject](../traces_glm/case_68/solo/r3/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r3/transcript.md) (correct) |
| r4 (new) | [Accept](../traces_glm/case_68/single_call/r4/transcript.md) (wrong) | [Reject](../traces_glm/case_68/solo/r4/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r4/transcript.md) (correct) |
| r5 (new) | [Reject](../traces_glm/case_68/single_call/r5/transcript.md) (correct) | [Reject](../traces_glm/case_68/solo/r5/transcript.md) (correct) | [Reject](../traces_glm/case_68/debate/r5/transcript.md) (correct) |

**The separation is intermittent.** In the three new trials, no tools was correct in 2/3, Solo in 3/3 and Debate in 3/3. Thus only one new trial reproduced the gap. Across all five trials, the corresponding counts are 2/5, 5/5 and 5/5, with no missing answers. Report both views: r1/r2 were the earlier selected successes, while r3–r5 were the fixed follow-up batch. These small counts do not establish a long-run success rate.

The no-tool successes have valid reasoning. In r3, the model corrected a missing factor of seven in its own calculation and derived that sufficiently large inputs with remainder six produce an oversized quotient. In r5, it also gave the legal witness `4294967291`: the kernel quotient is `613566756`, but exact division gives `613566755`. In r4, it miscalculated the multiplier as `2456838455` and the reciprocal excess as one rather than five, then incorrectly accepted. The initial API requests are identical across all five single-call trials; the result changes came from the responses, not changed prompts or additional hints.

All six new tool runs supplied valid executed counterexamples. Solo r4 initially assumed the division formula was correct, then tested all `2^32` scalar inputs on the GPU in legal-size batches and rejected after seeing failures. Its final report confused the first failing block start with the first failing input, so the actual witness must be read from probe stdout. Debate r4 also reached the correct verdict but misstated the multiplier and reference remainder, and its CPU failure count includes duplicate and out-of-domain values. Solo r5 also used a CPU checker whose int64 product overflowed, inflating its failure count; the model acknowledged that problem, and its separate GPU comparisons still establish rejection.

The semantic audit records each probe, input domain, reference, script repair and explanation limit in the [existing family manifest](private_data/source_fastdiv_pair.json). A correct verdict does not validate every associated claim: for example, no-tool r5 suggested increasing the shift to 35, but that proposed repair was not verified and would also require checking the wider multiplier for uint64 product overflow.

Tool initial requests differ from r1 only in bootstrap event timestamps; runtime records, source hashes, model settings and budgets match. Every raw request and response was captured. The tools consistently supplied executable rejection evidence across all five trials, but tools were not necessary for this model to solve the case: the two new no-tool derivations show that directly. There is still no observed accuracy advantage for Debate over Solo.

| New r3–r5 only | Correct | Wrong | Unfinished | API estimate |
|---|---:|---:|---:|---:|
| single_call | 2/3 | 1 | 0 | $0.018657 |
| solo | 3/3 | 0 | 0 | $0.599673 |
| debate | 3/3 | 0 | 0 | $0.759121 |

The nine new evaluations cost an estimated **$1.377451** in model API usage, excluding Modal GPU charges. Keep case 68 with the historically observed tool-benefit cases, but mark its separation **intermittent**, not consistently reproducible. The earlier 18-case-arm wave and its conditional r2 cost totals remain unchanged above.

## Atan, sigmoid and NTT

Six purpose-built implementations were qualified before model evaluation. Each has one fixed label over its complete public input domain. The pairs share the same requirements; their implementation code differs. Sources, failed CPU candidates, the reference repair and all model outcomes are retained.

All runs use Fireworks GLM-5p3, low reasoning effort, the unchanged verifier, a 32,768-output-token budget per arm, and the frozen T4 / PyTorch 2.8.0 / Triton 3.4.0 environment. The tool workflows receive only the current public case, not author-side oracle data.

### First pass

| Case | Implementation | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|---|
| [73](../triton_eval_cases/case_73/kernel.py) | Atan, five-digit coefficients | Accept | Accept ✓ | Accept ✓ | Accept ✓ |
| [74](../triton_eval_cases/case_74/kernel.py) | Atan, four-digit coefficients | Reject | Accept ✗ | Reject ✓ | Reject ✓ |
| [75](../triton_eval_cases/case_75/kernel.py) | Sigmoid, FP64 internal arithmetic | Accept | Accept ✓ | Accept ✓ | Accept ✓ |
| [76](../triton_eval_cases/case_76/kernel.py) | Sigmoid, FP32 internal arithmetic | Reject | Accept ✗ | Reject ✓ | Reject ✓ |
| [77](../triton_eval_cases/case_77/kernel.py) | NTT, converted fused coefficients | Accept | No verdict ✗ | Accept ✓ | Accept ✓ |
| [78](../triton_eval_cases/case_78/kernel.py) | NTT, unconverted fused coefficients | Reject | Accept ✗ | Reject ✓ | Reject ✓ |

Correct verdicts over all six first-pass cases: no tools **2/6**, Solo **6/6**, Debate **6/6**. Case 77's no-tool call exhausted all 32,768 output tokens without a verdict; it remains in the denominator and was not retried.

### Five fresh repetitions

The repeat rule was registered before model evaluation: a wrong no-tool first-pass verdict and valid correct verdicts from both tool workflows trigger exactly five fresh three-arm trials. All five must preserve that separation to pass this wave’s strict repeat check. Cases 74, 76 and 78 triggered the rule; all planned trials were completed, including negative results. These are selected follow-ups, reported separately from the first pass.

| Case | No tools correct | Solo correct | Debate correct | Five-repeat check |
|---|---|---|---|---|
| 74 | 2/5 | 5/5 | 5/5 | Did not pass |
| 76 | 2/5 | 5/5 | 4/5 | Did not pass |
| 78 | 0/5 | 5/5 | 5/5 | Passed |

Case 78 reproduced the separation in all five fresh trials: every no-tool answer accepted the defective kernel, and every Solo and Debate answer rejected it with an actual legal GPU counterexample. Including the initial selection trial, the counts are no tools 0/6 correct, Solo 6/6 and Debate 6/6. This is repeatability on one selected implementation and one model, not a measured success rate for all NTT kernels or evidence that Debate outperforms Solo.

### Why 78 separates the workflows

An NTT is an exact transform using integers modulo 3329. This implementation fuses two butterfly stages and stores multiplication coefficients in Montgomery form: each stored coefficient already includes a scale factor R. The faulty version multiplies two stored coefficients without removing one extra factor of R. The individual butterfly formulas and individual modular-reduction operation look valid, but their combination computes the wrong transform. This is a synthetic porting defect inspired by the Montgomery representation used in real NTT implementations, not a claimed upstream historical bug.

The following is an explanatory sketch of the coefficient conversion:

```python
a_R = a * R % q
b_R = b * R % q
cross = a_R * b_R % q             # Contains R squared.
cross_R = cross * R_inverse % q   # Restore the one-R representation.
```

The no-tool model repeatedly accepted the composition. In r3 it even identified the R-squared versus R distinction, then incorrectly treated them as equivalent. The tool workflows executed the actual kernel and compared its output with an independent exact integer DFT. A simple recorded counterexample is N=64 with all inputs equal to 1: the zero-frequency output must be the sum, **64**, but the kernel returns **2954**. This legal counterexample supports the implementation’s fixed Reject label; the input itself does not receive a label. See [Solo r3](../traces_glm/case_78/solo/r3/transcript.md), [No tools r3](../traces_glm/case_78/single_call/r3/transcript.md) and the [full NTT audit](private_data/source_ntt_pair.json).

### What the other two directions showed

Atan 74 uses four-significant-digit coefficients in a Cephes-style rational approximation. Independent execution finds 5,606 failing finite FP16 encodings, with maximum absolute error about 5.033e-6 against a 1e-6 requirement. Wrong no-tool answers transferred the original approximation’s accuracy to this shortened coefficient set. Two fresh no-tool runs nevertheless rejected it; one spent 16,954 output tokens on manual arithmetic, while another reached the right label using incorrectly recalled source coefficients. All tool repetitions found actual violations. The gap is intermittent, so this case did not pass the strict repeat check.

Sigmoid 76 has an explicit requirement to round the mathematical result once to FP16. This is stricter than ordinary approximate sigmoid API behavior. Its FP32 path misrounds seven finite input encodings; for x=0.0029296875, the correct output is 0.50048828125, while the kernel returns 0.5009765625. Some no-tool runs dismissed rounding-boundary crossings as too unlikely; two fresh runs correctly rejected the implementation. The gap is intermittent here as well. Debate r3 abstained after repeated input-enumeration errors and a blocked corrective probe; it had budget remaining but never completed a valid near-zero check. This is a workflow failure, not a GPU or API outage. Case 75’s FP64 implementation passes the same complete scalar-domain requirement.

### Qualification and trace limitations

Atan and sigmoid qualification covers all 63,488 finite FP16 encodings with independent directed MPFR bounds and a second high-precision implementation. NTT qualification compares the complete basis operator with a Python big-integer DFT and separately verifies modular linearity, integer bounds and address layouts; it therefore does not rely only on randomly passing inputs. Compiled GPU runs also check storage and shape cases. These checks are evidence for the fixed implementations and environment, not an exhaustion of every possible tensor execution.

The first sigmoid qualification attempt exposed a platform-dependent FP64-to-FP16 conversion in the public reference. Before any model call on those cases, the reference was repaired to write FP16 bits directly with Python struct.pack. The mathematical requirement, GPU kernel and expected labels stayed unchanged. The failed [r22 qualification](private_data/qualification/r22.json), old hashes and repair are retained; the repaired pair passed [r24](private_data/qualification/r24.json). Atan passed [r21](private_data/qualification/r21.json); NTT passed [r23](private_data/qualification/r23.json).

Trace review distinguished real executed counterexamples from failed scripts and inflated coverage claims. Examples include FP32 nextafter values collapsing to duplicate FP16 values, invalid reference dtypes, an incorrect sigmoid helper, and NTT sample tests described as complete-domain coverage. Those statements are not counted as additional evidence. Every Reject from a tool workflow counted in the repetition table has a separate valid actual GPU witness. The correct NTT control’s tool runs use valid samples but do not themselves reproduce the author-side complete-basis proof.

Total: **63 model workflow trials**, including 18 first-pass trials and 45 selected repetitions. Recorded model API estimate: **$11.208477**; first pass **$2.784000**, repetitions **$8.424477**. This is a conservative estimate from captured usage and the recorded price schedule, not an invoice; Modal GPU charges are excluded. All raw requests, responses and tool artifacts remain in [traces_glm](../traces_glm/INDEX.md).

Per-family records: [Atan](private_data/source_atan_pair.json), [sigmoid rounding](private_data/source_rounding_pair.json), [NTT](private_data/source_ntt_pair.json). These include predeclared plans, source hashes, every result, protocol comparisons and per-run semantic audits.

## CRC, complex division and INT8 requantization

This wave tests three directions with six purpose-built kernels, cases 79–84. Each pair has one correct implementation and one defective implementation under the same public requirements. Each kernel keeps one label over the full permitted input domain. The implementations adapt real arithmetic methods; these are synthetic variants, not reported upstream bugs.

All six kernels passed independent qualification before model calls. All three arms use Fireworks GLM-5p3 with low reasoning effort and a 32,768-output-token budget. The tool workflows use the unchanged verifier on NVIDIA T4, PyTorch 2.8.0 and Triton 3.4.0. Private qualification data are not shown to the model.

### First pass

| Case | Implementation | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|---|
| [79](../triton_eval_cases/case_79/kernel.py) | CRC with the correct reflected unit | Accept | [Accept](../traces_glm/case_79/single_call/r1/transcript.md) ✓ | [Accept](../traces_glm/case_79/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_79/debate/r1/transcript.md) ✓ |
| [80](../triton_eval_cases/case_80/kernel.py) | CRC with the wrong reflected unit | Reject | [Reject](../traces_glm/case_80/single_call/r1/transcript.md) ✓ | [Reject](../traces_glm/case_80/solo/r1/transcript.md) ✓ | [Reject](../traces_glm/case_80/debate/r1/transcript.md) ✓ |
| [81](../triton_eval_cases/case_81/kernel.py) | Complex division with direct FP32 products | Accept | [Reject](../traces_glm/case_81/single_call/r1/transcript.md) ✗ | [Accept](../traces_glm/case_81/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_81/debate/r1/transcript.md) ✓ |
| [82](../triton_eval_cases/case_82/kernel.py) | Complex division with a rounded ratio | Reject | [Reject](../traces_glm/case_82/single_call/r1/transcript.md) ✓ | [Reject](../traces_glm/case_82/solo/r1/transcript.md) ✓ | [Reject](../traces_glm/case_82/debate/r1/transcript.md) ✓ |
| [83](../triton_eval_cases/case_83/kernel.py) | INT8 rescale with exact int64 arithmetic | Accept | [Accept](../traces_glm/case_83/single_call/r1/transcript.md) ✓ | [Accept](../traces_glm/case_83/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_83/debate/r1/transcript.md) ✓ |
| [84](../triton_eval_cases/case_84/kernel.py) | INT8 rescale through FP64 arithmetic | Reject | [Reject](../traces_glm/case_84/single_call/r1/transcript.md) ✓ | [Reject](../traces_glm/case_84/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_84/debate/r1/transcript.md) ✗ |

First-pass correct verdicts: **no tools 5/6, Solo 6/6, Debate 5/6**. All 18 trials produced a verdict. Case 81 showed a first-pass tool benefit. Case 84 showed a Debate failure. Both results remain in the totals.

### 79–80: combining CRC blocks

These kernels compute CRC-32 over 64-byte and 128-byte rows. They combine four-byte blocks with polynomial arithmetic over bits. In this reflected representation, the multiplicative unit is `0x80000000`. Case 80 uses `1` in the exponentiation helper, so block composition computes the wrong checksum. Case 79 uses the correct unit.

Independent qualification compares the complete affine basis with a separate bitwise recurrence and zlib. It also checks integer ranges and addresses. The actual GPU checks cover 2,284 outputs per implementation. Case 79 has no mismatch; case 80 has 2,272. Passing zero rows does not mean that every nonzero row fails: different messages can have the same CRC.

All three arms identify both labels. The no-tool model uses 15,176 output tokens on case 80 and derives the representation error from source. Tool execution adds reproducible failures, but this pair adds no verdict accuracy gain. [Source and trace audit](private_data/source_crc_pair.json).

### 81–82: complex division

The requirement is complex division for finite FP16 components with a nonzero denominator. Each FP32 output component must satisfy `absolute error <= 1e-7 + 5e-7 * abs(target)`. Case 81 uses the direct formula (equivalent short form):

```python
scale = c*c + d*d
real = tl.div_rn(a*c + b*d, scale)
imag = tl.div_rn(b*c - a*d, scale)
```

The inputs are FP16, but these operations use FP32. Each FP16 product needs at most 22 significant bits, so FP32 represents it exactly. The subsequent sums and divisions round. Their relative error bound is about `1.78814e-7`, below `5e-7`; the permitted input range prevents FP32 overflow and underflow. The argument covers the scalar arithmetic. Separate GPU checks cover shapes and storage conditions.

The first no-tool answer rejects case 81 because it assumes that the products already lost precision before cancellation. It also uses `65503`, which is not a representable FP16 value, in a proposed example. Solo and Debate accept after actual GPU comparisons. The independent qualification checks 937,928 output components with no failure.

Case 82 divides the smaller denominator component by the larger one, then uses this rounded ratio in fused operations. Ratio error can remain when terms cancel. For the legal input `(1024 - 1536i) / (3 + 2i)`, the exact real component is zero. The actual GPU result is about `-7.04252e-6`. Its absolute error exceeds `1e-7`. Fused multiply-add does not restore the exact ratio. All three arms reject this implementation. [Source, arithmetic bound and trace audit](private_data/source_complex_pair.json).

### Case 81: five fresh trials

Before any model call, the repeat rule specified five fresh three-arm trials for a qualifying first-pass gap. Only case 81 qualified. All five trials ran with the same source, public requirements, model, instructions and budgets. Initial tool requests differ only in bootstrap timestamps; single-call requests are identical. The table excludes the initial selection trial.

| Trial | No tools | Solo | Debate |
|---|---|---|---|
| r2 | [Accept](../traces_glm/case_81/single_call/r2/transcript.md) ✓ | [Accept](../traces_glm/case_81/solo/r2/transcript.md) ✓ | [Accept](../traces_glm/case_81/debate/r2/transcript.md) ✓ |
| r3 | [Accept](../traces_glm/case_81/single_call/r3/transcript.md) ✓ | [Accept](../traces_glm/case_81/solo/r3/transcript.md) ✓ | [Accept](../traces_glm/case_81/debate/r3/transcript.md) ✓ |
| r4 | [Accept](../traces_glm/case_81/single_call/r4/transcript.md) ✓ | [Accept](../traces_glm/case_81/solo/r4/transcript.md) ✓ | [Accept](../traces_glm/case_81/debate/r4/transcript.md) ✓ |
| r5 | [Accept](../traces_glm/case_81/single_call/r5/transcript.md) ✓ | [Accept](../traces_glm/case_81/solo/r5/transcript.md) ✓ | [Accept](../traces_glm/case_81/debate/r5/transcript.md) ✓ |
| r6 | [Accept](../traces_glm/case_81/single_call/r6/transcript.md) ✓ | [Accept](../traces_glm/case_81/solo/r6/transcript.md) ✓ | [Accept](../traces_glm/case_81/debate/r6/transcript.md) ✓ |

Fresh-trial correct counts: **no tools 5/5, Solo 5/5, Debate 5/5**. The first-pass gap did not pass the five-repeat requirement. In the fresh no-tool answers, the model recognizes that FP16 products are exact in FP32. In r3, it reaches this conclusion in only 581 output tokens. Keep case 81 as an observed first-pass benefit with failed replication, not a stable new separation.

### 83–84: exact INT8 rescale

These kernels rescale int32 accumulators with a per-channel multiplier, shift and zero point. The output must equal the exact integer rule, including its specified tie handling:

```python
P = a * multiplier
R = (P + 2**(30 + shift)) // 2**(31 + shift)
output = clamp(R + zero_point, -128, 127)
```

Case 83 performs the operation in int64. Range analysis proves that the product plus bias fits, and that its intermediate clamp preserves the final INT8 output. Case 84 uses FP64 for the product and rounding step. An exact integer product can need more than FP64's 53 significant bits.

A legal failing example uses `a=1073741825`, `multiplier=2147483646`, `shift=31` and `zero_point=0`. The exact product is `2**61 - 2`; the correct output is **0**. FP64 rounds the product to `2**61`, and the kernel returns **1**. This error survives INT8 saturation. The independent GPU checks find 2,137 failing outputs in case 84 and none in case 83.

Solo finds 16 actual GPU failures and correctly rejects case 84. Debate accepts after 40,164 matching output comparisons. Its main precision probes use large accumulators with shift zero: both outputs reach the same INT8 saturation limit, which hides the rounding error. Its remaining coarse tests miss the unsaturated half-integer boundary at large shifts. Passing those tests does not establish correctness.

The no-tool verdict on case 84 is correct, but its stated example uses shift 30 and incorrectly claims that `floor(1.5)` is 2. Both implementations return 1 for that example. Count the verdict as correct and record the invalid explanation separately. [Source and full trace audit](private_data/source_requant_compose_pair.json).

### Evidence limits and next direction

Every trial has raw requests, responses, usage, verdicts and tool artifacts. Some failed scripts were repaired; the failures remain in the traces. Some tool summaries overstate coverage. For example, a case 81 cancellation generator falls back to samples that do not meet its intended near-cancellation condition. Other probes lose storage offsets when copying inputs, or check only finiteness while claiming accuracy coverage. The semantic audits identify these errors. Passing samples support only the tested conditions; they do not replace the independent arithmetic argument.

This wave found no new stable separation between no tools and both tool workflows. More source complexity alone did not help: the no-tool model derived the CRC composition rule. A single numerical reasoning error also did not persist: case 81 was corrected in fresh trials. Tools supplied valid counterexamples, but poor test selection caused the case 84 Debate error. This wave does not show a Debate advantage over Solo.

The next candidates are fused transform stages with interacting scale and index conversions, and parallel affine scans with ordered matrix composition. Complete basis checks can verify exact linear operators; floating-point variants also need an error argument. These directions are proposals, not results from this wave.

Total: **33 workflow trials**: 18 first-pass trials and 15 selected repetitions. The conservative model API estimate is **$6.476464**: first pass **$3.629795**, repetitions **$2.846669**. The estimate uses captured usage and the recorded Fireworks prices; it is not an invoice and excludes Modal GPU charges. Independent qualification: [CRC](private_data/qualification/r25.json), [complex division and INT8](private_data/qualification/r26.json). All trial paths are listed in the [GLM index](../traces_glm/INDEX.md).

## Affine scan, Winograd and real FFT

This wave adds six kernels in three pairs, cases 85–90. Each pair uses the same public requirements and different code. The correct code has one Accept label. The defective code has one Reject label. Legal inputs provide evidence about the implementation; they do not receive separate labels. These are synthetic adaptations of real algorithms, not reproductions of historical upstream bugs.

Independent T4 checks confirmed each fixed label before model evaluation. All three methods use Fireworks GLM-5p3 with low reasoning effort. Each trial has a 32,768-output-token budget. Solo and the existing four-role Debate use the unchanged verifier, PyTorch 2.8.0 and Triton 3.4.0. All raw API requests, responses, usage, verdicts and tool artifacts are retained.

### First pass

| Case | Implementation | Fixed label | No tools | Solo | Debate |
|---|---|---|---|---|---|
| [85](../triton_eval_cases/case_85/kernel.py) | Affine scan with ordered composition | Accept | [Accept](../traces_glm/case_85/single_call/r1/transcript.md) ✓ | [Accept](../traces_glm/case_85/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_85/debate/r1/transcript.md) ✓ |
| [86](../triton_eval_cases/case_86/kernel.py) | Affine scan with reversed carry composition | Reject | [Accept](../traces_glm/case_86/single_call/r1/transcript.md) ✗ | [Reject](../traces_glm/case_86/solo/r1/transcript.md) ✓ | [Reject](../traces_glm/case_86/debate/r1/transcript.md) ✓ |
| [87](../triton_eval_cases/case_87/kernel.py) | Winograd with consistent transform coordinates | Accept | [Accept](../traces_glm/case_87/single_call/r1/transcript.md) ✓ | [Accept](../traces_glm/case_87/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_87/debate/r1/transcript.md) ✓ |
| [88](../triton_eval_cases/case_88/kernel.py) | Winograd with swapped filter coordinates | Reject | [Reject](../traces_glm/case_88/single_call/r1/transcript.md) ✓ | [Reject](../traces_glm/case_88/solo/r1/transcript.md) ✓ | [Reject](../traces_glm/case_88/debate/r1/transcript.md) ✓ |
| [89](../triton_eval_cases/case_89/kernel.py) | Real FFT with the correct partner address | Accept | [Accept](../traces_glm/case_89/single_call/r1/transcript.md) ✓ | [Accept](../traces_glm/case_89/solo/r1/transcript.md) ✓ | [Accept](../traces_glm/case_89/debate/r1/transcript.md) ✓ |
| [90](../triton_eval_cases/case_90/kernel.py) | Real FFT with the wrong partner address | Reject | [Accept](../traces_glm/case_90/single_call/r1/transcript.md) ✗ | [Reject](../traces_glm/case_90/solo/r1/transcript.md) ✓ | [Reject](../traces_glm/case_90/debate/r1/transcript.md) ✓ |

First-pass correct verdicts: **no tools 4/6, Solo 6/6, Debate 6/6**. All six cases remain in this table. Follow-up repetitions are reported separately.

### 85–86: ordered affine scan

The task updates a two-component state at each step: `state = matrix @ state + bias`. The kernel must return every state exactly. Sequence lengths range from 1 to 32. Matrix entries are −1, 0 or 1; bias and initial state use the full INT16 range. Output uses INT64. The implementation first scans groups of eight steps, then combines their summaries and applies the incoming state to each group.

Case 86 reverses two arguments when it propagates a prefix through the summary tree. The helper applies its right argument after its left argument. The incorrect call applies the parent prefix after the left subtree, which reverses the required order. The error becomes visible in the fourth group, at lengths 25–32. Shorter sequences and some commuting transforms can pass. The implementation still has one Reject label for the full requirements.

A small independent witness uses a 25-step sequence. All steps are identity transforms with zero bias, except `[[1,1],[0,1]]` at step 0 and `[[1,0],[1,1]]` at step 16. From initial state `[1, 2]`, the correct state at step 24 is `[3, 5]`. Case 86 returns `[4, 3]`. Case 85 uses the correct argument order. Independent qualification compares 53,780 components per case with a sequential Python-integer reference. Case 85 passes all 119 plans. Case 86 fails 28 plans, with 3,310 incorrect components. Composition, address and integer-range checks support the control over the full declared domain.

The first no-tool answer for case 86 derives the helper's composition rule correctly, but accepts the wrong downsweep order. It checks the familiar addition-scan pattern without a correct invariant for a nonidentity parent prefix. The tool workflows can compare the complete recurrence against a sequential reference. Their actual evidence and coverage limits are recorded in the [source and trace audit](private_data/source_affine_scan_pair.json).

### 87–88: Winograd convolution

Cases 87–88 compute a valid 3×3 cross-correlation for each 6×6 INT8 tile. Each tile has its own filter. Every INT8 value is legal, and the 4×4 INT32 output must be exact. Both cases have the same public requirements. Case 87 is Accept. Case 88 is Reject. The labels apply to each complete implementation.

The kernels use Winograd transforms to compute the cross-correlation. The filter transform uses coordinates that include +2 and −2. Case 88 swaps these two coordinates, but keeps the data and output transforms unchanged. The products therefore combine different coordinates. Case 87 keeps the same coordinate order in all three stages.

Independent qualification checked all 324 tile/filter basis pairs and bounded every integer intermediate. These checks establish the complete arithmetic identity for case 87. On T4, all 130,752 tested outputs passed. Case 88 had 125,623 failed outputs. The GPU plans also checked tail blocks, storage offsets, overlapping inputs, input preservation and separate output storage.

All three methods gave the correct verdict for both cases. The no-tool model derived a valid transform identity for case 87. For case 88, it found the coordinate mismatch but calculated its final counterexample incorrectly. Both tool methods found valid GPU failures. Some trace summaries overstated test coverage or reported incorrect failure counts. This pair shows no verdict advantage from tools.

[Winograd source and trace audit](private_data/source_winograd_pair.json).

### 89–90: packed real FFT

The task computes the unnormalized forward Fourier transform of each real FP16 row. Legal widths are 64, 128 and 256. Every finite FP16 value is allowed. The output stores FP32 real and imaginary components for frequencies `k=0..N/2` in natural order. Each component must satisfy `absolute error <= 1e-4 + 1e-6 * abs(target)`.

Let `M = N/2`. The kernel packs even and odd input positions into a half-length complex transform. Its internal result is stored in bit-reversed order. The final real-transform step needs both frequency `k` and frequency `(-k) mod M`. It must convert each frequency to its storage address. The two implementations differ here (simplified sketch; the actual reversal helper also receives the bit width):

```python
# Case 89: correct. Convert the required partner frequency to an address.
partner = reverse((-k) & (M - 1))

# Case 90: defective. Negate the address of k instead.
address = reverse(k)
partner = (-address) & (M - 1)
```

Bit reversal and modular negation do not generally commute. For `M=32` and `k=1`, the correct partner address is 31. Case 90 uses 16. It therefore combines the wrong complex coefficients. FP64 arithmetic cannot correct this address error.

Independent qualification uses a separate direct DFT with bounded reference error, all impulse positions and additional boundary checks. Case 89 passes all 36 GPU plans. Case 90 fails 24 plans, with 355,421 failed output components. The control also has an arithmetic error bound: at most `2.383022e-5 + 5.960465e-8 * abs(target)`, below the public tolerance. Basis tests alone do not prove a floating-point error bound.

The first no-tool answer for case 90 checks the butterfly and real-transform formulas but assumes that the partner coefficient is correct. It accepts the kernel. Solo first also considers the formula correct, then rejects after actual GPU failures. Debate also finds legal GPU counterexamples. Some initial extreme-value probes create infinity when converted to FP16. Those inputs are illegal and are excluded as evidence. Separate finite random inputs, tones and impulses still establish the defect. [FFT source and trace audit](private_data/source_realfft_pair.json).

### Five fresh trials for each selected case

Before model evaluation, the plan specified five fresh three-arm trials for each qualifying first-pass gap, with at most three selected cases. Selection requires a wrong no-tool verdict and correct verdicts from both tool workflows with legal actual GPU evidence. Every selected trial runs, even if the gap disappears. Code, public requirements, prompts, runtime and budgets stay fixed. The following results exclude the first-pass selection trials.


#### case_86

| Trial | No tools | Solo | Debate |
|---|---|---|---|
| r2 | [Reject](../traces_glm/case_86/single_call/r2/transcript.md) ✓ | [Reject](../traces_glm/case_86/solo/r2/transcript.md) ✓ | [Reject](../traces_glm/case_86/debate/r2/transcript.md) ✓ |
| r3 | [Accept](../traces_glm/case_86/single_call/r3/transcript.md) ✗ | [Reject](../traces_glm/case_86/solo/r3/transcript.md) ✓ | [Reject](../traces_glm/case_86/debate/r3/transcript.md) ✓ |
| r4 | [Accept](../traces_glm/case_86/single_call/r4/transcript.md) ✗ | [Reject](../traces_glm/case_86/solo/r4/transcript.md) ✓ | [Abstain](../traces_glm/case_86/debate/r4/transcript.md) ✗ |
| r5 | [Reject](../traces_glm/case_86/single_call/r5/transcript.md) ✓ | [Reject](../traces_glm/case_86/solo/r5/transcript.md) ✓ | [Reject](../traces_glm/case_86/debate/r5/transcript.md) ✓ |
| r6 | [Accept](../traces_glm/case_86/single_call/r6/transcript.md) ✗ | [Reject](../traces_glm/case_86/solo/r6/transcript.md) ✓ | [Reject](../traces_glm/case_86/debate/r6/transcript.md) ✓ |

Fresh correct counts: **no tools 2/5, Solo 5/5, Debate 4/5**. This case does not meet the predeclared five-repeat separation check.

#### case_90

| Trial | No tools | Solo | Debate |
|---|---|---|---|
| r2 | [Accept](../traces_glm/case_90/single_call/r2/transcript.md) ✗ | [Reject](../traces_glm/case_90/solo/r2/transcript.md) ✓ | [Reject](../traces_glm/case_90/debate/r2/transcript.md) ✓ |
| r3 | [Reject](../traces_glm/case_90/single_call/r3/transcript.md) ✓ | [Reject](../traces_glm/case_90/solo/r3/transcript.md) ✓ | [Reject](../traces_glm/case_90/debate/r3/transcript.md) ✓ |
| r4 | [Accept](../traces_glm/case_90/single_call/r4/transcript.md) ✗ | [Reject](../traces_glm/case_90/solo/r4/transcript.md) ✓ | [Reject](../traces_glm/case_90/debate/r4/transcript.md) ✓ |
| r5 | [Reject](../traces_glm/case_90/single_call/r5/transcript.md) ✓ | [Reject](../traces_glm/case_90/solo/r5/transcript.md) ✓ | [Reject](../traces_glm/case_90/debate/r5/transcript.md) ✓ |
| r6 | [Accept](../traces_glm/case_90/single_call/r6/transcript.md) ✗ | [Reject](../traces_glm/case_90/solo/r6/transcript.md) ✓ | [Reject](../traces_glm/case_90/debate/r6/transcript.md) ✓ |

Fresh correct counts: **no tools 2/5, Solo 5/5, Debate 5/5**. This case does not meet the predeclared five-repeat separation check.

Including the first pass, case_86 has correct counts of **no tools 2/6, Solo 6/6, Debate 5/6**. The difference is intermittent. It does not pass the five-repeat requirement.

Including the first pass, case_90 has correct counts of **no tools 2/6, Solo 6/6, Debate 6/6**. The difference is intermittent. It does not pass the five-repeat requirement.

Case 86 no-tools r2 correctly derives the reversed operands and explains why the bad carry first matters at length 25. In r3, the model instead writes one composition order and treats it as another. The same source therefore produces both correct and incorrect manual reviews. The tool runs can execute the full sequence and compare every state with the simple sequential reference.

Case 86 Debate r4 abstains without running a GPU probe. Two early role calls reach the per-call output limit. A later call records a claim, but the run enters the reserved closeout before execution. It uses 25,261 output tokens in total. This is a reasoning-budget and scheduling failure, not an API or GPU outage. It remains in the denominator and is not counted as correct. Solo supplies valid executed counterexamples in the corresponding trials.

Case 90 no-tools r2 explicitly claims that bit reversal commutes with modular negation. It incorrectly treats negation as bitwise complement and omits the added one. In r3, the model eventually identifies the true address error. Its final numerical example uses N=8, which is outside the legal widths. The source diagnosis is correct, so the verdict counts as correct; the example is not a legal benchmark counterexample. This distinction is retained in the audit.

Case 90 no-tools r5 also reaches the correct verdict and checks a legal M=32 address mismatch. Its final N=8 numerical example remains outside the domain and contains a packing error: z=[1,i,0,0] has exact frequency-1 real part about 0.2929, not 1.707. The verdict is still counted as correct. No invalid hand example is presented as a valid executed counterexample.

Case 90 Solo r2 and Debate r2 initially accept the index argument, then change their verdict after GPU comparisons fail. Debate r2 does not correctly isolate the cause: it blames Triton gather behavior without a test that establishes that explanation. The executed mismatch supports Reject; the source address error explains the defect. A correct verdict does not make every explanation correct.

These results support further work on the boundaries between algorithm stages: transform-coordinate order, physical storage order and noncommutative composition. Extra code length alone is not the objective. Tools can enumerate address mappings, check symbolic composition or compare a complete transform basis, as well as run numeric samples. The current results compare complete workflows. They do not isolate tool access from extra reasoning, and they do not establish a Debate advantage over Solo.

### Qualification, evidence and cost

The first affine GPU qualification failed to compile because a summary store used a scalar pointer with a vector value. Both implementations received the same pointer-shape repair before any model call. The expected labels and public requirements stayed fixed. The failed [r28 record](private_data/qualification/r28.json), old hashes and repair are retained. The repaired pair passed [r29](private_data/qualification/r29.json). The FFT cases already passed their part of r28. Their completed evidence was recorded separately after hash verification. Winograd passed [r27](private_data/qualification/r27.json).

Trace review separates correct verdicts from correct explanations. Some models overstate sample coverage or quote the wrong failure count. Failed scripts and illegal probes remain in the traces, but do not count as evidence. The private manifests contain the per-trial semantic review and the comparison of repeated initial requests. Repeated single-call requests are identical; tool requests differ only in bootstrap timestamps. All repeated runtime settings match.

Two concurrent no-tool batches, r2 and r3, received the same timestamp batch ID. Their separate trial directories and different API response IDs confirm independent calls. The index now separates these collisions by the canonical trial number and preserves the original metadata. Future batch IDs also include a random suffix. These reporting and storage changes were made after all model trials finished.

Total: **48 workflow trials**, including 18 first-pass trials and 30 selected repetitions. The conservative model API estimate is **$9.275751**: first pass **$3.548831**, repetitions **$5.726920**. This estimate uses captured token usage and the recorded price schedule. It is not an invoice and excludes Modal GPU charges. All trials are listed in the [GLM trace index](../traces_glm/INDEX.md).

## October 20: exp2, log1p and AES

These six cases test whether tools help a model make the correct decision. They do not establish an advantage of Debate over Solo. Each pair has the same problem description and one fixed label per implementation. Test inputs provide evidence for that label. They do not change the label.

All cases take a one-dimensional contiguous CUDA tensor with any length from 1 through 131072. Nonzero storage offsets are legal. Each result must use fresh storage, preserve the input and have the same shape and device as the input. Numerical results below use NVIDIA T4, PyTorch 2.8.0 and Triton 3.4.0. Cases 163–166 require ordinary tensors with no gradient, inference, negative or conjugate flags, in ordinary execution mode without concurrent input writes. The AES pair also supports lazy-negative views.

### 163–164: exp2 coefficients

**What it should do.** For every finite BF16 value in `[-10, 10]`, compute `2 ** x` and return FP32. The allowed relative error is `1e-6`; the absolute tolerance is zero. Both signed zeros and subnormal values are legal.

**What changes.** Both kernels write `x = integer + fraction`, approximate `2 ** fraction` with the same degree-five polynomial structure, then multiply by `2 ** integer`. Case 163 stores more rounded polynomial coefficients. Case 164 retains more digits. For example, one coefficient changes from `0.69315` to `0.693147`. The target and tolerance stay the same.

**What the check found.** Case 163 is **Reject**. At `x = 0.5`, its output is `1.4142098426818848`, while the target is approximately `1.4142135623730951`. The relative error is about `2.63e-6`, above the limit. Across all 33,346 legal BF16 encodings, 455 fail. Case 164 is **Accept**: its maximum relative error bound is `2.257e-7`, below the same limit.

**Why tools may help.** Both implementations have the correct range reduction and polynomial structure. Coefficient rounding, polynomial evaluation and FP32 rounding together determine the final error. The number of coefficient digits alone does not establish whether the implementation meets the limit. Tools can check every legal scalar value against an independent mathematical reference.

### 165–166: log1p range reduction

**What it should do.** For every finite FP16 value greater than `-1`, compute `log(1 + x)` and return FP32. The allowed absolute error is `1e-6`; the relative tolerance is zero. The domain includes subnormal values, the FP16 value immediately above `-1`, and `65504`.

**What changes.** Both kernels form `u = 1 + x`, split `u` into a power of two and a mantissa, evaluate a short polynomial, then restore the exponent contribution. A correction term accounts for rounding when forming `u`. Case 165 halves the mantissa when it exceeds `1.5`. Case 166 changes this threshold to `1.65`, while keeping the polynomial and all other steps unchanged.

```python
upper = mantissa > threshold
mantissa = where(upper, mantissa * 0.5, mantissa)
exponent = exponent + upper
z = (mantissa - 1.0) / (mantissa + 1.0)
```

**What the check found.** Case 165 is **Accept**. Across all 47,104 legal FP16 encodings, its maximum absolute error bound is `6.675e-7`. Case 166 is **Reject**: 55 encodings exceed the limit. At `x = 3378`, the output is `8.125333786010742`, while the target is approximately `8.12533508671429`. The absolute error is about `1.301e-6`.

**Why tools may help.** Changing the threshold preserves the exact mathematical decomposition, but changes the interval on which the polynomial must work. The reconstruction and its FP32 rounding also affect the result. A familiar formula does not establish that the final error remains below `1e-6`. An exhaustive scalar check can determine that boundary.

### 167–168: AES field conversion

**What it should do.** Apply the AES SubBytes transformation to every uint8 value, with exact uint8 output. Every byte from 0 to 255 is legal in every arrangement. The target uses the AES polynomial `0x11b`, multiplicative inversion with zero mapped to zero, and the standard affine transformation with constant `0x63`. For example, input `0x53` must produce `0xed`.

**What changes.** The implementation converts each byte into two four-bit components, computes the inverse in that representation, then converts back and applies the affine transformation. Case 167 uses the extension equation `u² + u + 8 = 0`, which matches its conversion matrices. Case 168 uses `u² + u + 9 = 0` with the same matrices. That change defines a different multiplication rule. The conversion matrices remain invertible, but they no longer preserve AES multiplication.

**What the check found.** The complete scalar audit gives **Accept** for case 167 and **Reject** for case 168. Case 168 differs from AES on 240 of the 256 byte values. Input `2` produces `46`; the required output is `119`. Both implementations still produce 256 distinct outputs. Checking that the output is a permutation would therefore miss the defect. T4 qualification after the repair confirms these fixed labels; see [r31](private_data/qualification/r31.json).

**Why tools may help.** The inversion formula and the conversion matrices can each look valid in isolation. Correctness also requires them to use the same field representation. Tools can compare all 256 outputs with an independent AES definition and check all 65,536 multiplication pairs. The checks identify a representation mismatch without relying on an assumption that valid-looking components compose correctly.

**Pre-evaluation repair.** An independent review found that the original shared wrapper read physical storage bytes from lazy-negative tensor views. Both wrappers now call `resolve_neg()` before launching Triton. The problem description did not change. The original qualification is retained, and the repaired versions passed new GPU checks that include the full byte domain, lazy-negative views, offsets and backing-storage preservation. No model result was used to select this repair.

These are synthetic implementations of established operations, not reproductions of historical upstream bugs. Complete scalar checks and separate storage checks support the labels within the stated environment. They do not exhaust every possible tensor execution or GPU platform. The three model workflows use the protocol in [the experiment record](private_data/experiment_20261020.json). The results and trace analysis are recorded below.

<!-- BEGIN REAL KERNEL OCTOBER20 RESULTS -->

以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。
正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。

| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |
|---|---|---|---|---|
| case_163 | reject | [r1](../traces_glm/case_163/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0252 | [r1](../traces_glm/case_163/solo/r1/transcript.md)：reject / 正确；probe 1；$0.0901 | [r1](../traces_glm/case_163/debate/r1/transcript.md)：reject / 正确；probe 4；$0.2675 |
| case_164 | trust | [r1](../traces_glm/case_164/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0098 | [r1](../traces_glm/case_164/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0744 | [r1](../traces_glm/case_164/debate/r1/transcript.md)：trust / 正确；probe 4；$0.2397 |
| case_165 | trust | [r1](../traces_glm/case_165/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0117 | [r1](../traces_glm/case_165/solo/r1/transcript.md)：trust / 正确；probe 2；$0.1012 | [r1](../traces_glm/case_165/debate/r1/transcript.md)：trust / 正确；probe 2；$0.1881 |
| case_166 | reject | [r1](../traces_glm/case_166/single_call/r1/transcript.md)：trust / 错误；probe 0；$0.0084<br>[r2](../traces_glm/case_166/single_call/r2/transcript.md)：trust / 错误；probe 0；$0.0097<br>[r3](../traces_glm/case_166/single_call/r3/transcript.md)：reject / 正确；probe 0；$0.0463 | [r1](../traces_glm/case_166/solo/r1/transcript.md)：reject / 正确；probe 2；$0.1305<br>[r2](../traces_glm/case_166/solo/r2/transcript.md)：reject / 正确；probe 1；$0.0784<br>[r3](../traces_glm/case_166/solo/r3/transcript.md)：reject / 正确；probe 1；$0.1085 | [r1](../traces_glm/case_166/debate/r1/transcript.md)：reject / 正确；probe 3；$0.2650<br>[r2](../traces_glm/case_166/debate/r2/transcript.md)：reject / 正确；probe 4；$0.5787<br>[r3](../traces_glm/case_166/debate/r3/transcript.md)：reject / 正确；probe 3；$0.2675 |
| case_167 | trust | [r1](../traces_glm/case_167/single_call/r1/transcript.md)：trust / 正确；probe 0；$0.0449 | [r1](../traces_glm/case_167/solo/r1/transcript.md)：trust / 正确；probe 1；$0.0814 | [r1](../traces_glm/case_167/debate/r1/transcript.md)：trust / 正确；probe 3；$0.2025 |
| case_168 | reject | [r1](../traces_glm/case_168/single_call/r1/transcript.md)：reject / 正确；probe 0；$0.0896 | [r1](../traces_glm/case_168/solo/r1/transcript.md)：reject / 正确；probe 4；$0.4095 | [r1](../traces_glm/case_168/debate/r1/transcript.md)：reject / 正确；probe 5；$0.3469 |

| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_call | 8 | 6 | 2 | 0 | 0 | 0 | 0 | $0.2456 |
| solo | 8 | 8 | 0 | 0 | 0 | 0 | 13 | $1.0738 |
| debate | 8 | 8 | 0 | 0 | 0 | 0 | 28 | $2.3559 |

目标 **18 个 case×arm 槽位**：已有记录 18，缺失 0。共有 24 次尝试；按互斥结果类别计：其他运行失败 0，弃答 0，其他无最终答案 0，输出截断／上限 0，未完成 0。
全部尝试的已记录 API 费用估算：**$3.6753**；费用未知 0 次、usage 不完整 0 次。优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。

实际启动模型的评测有 **24 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：

| Arm | 判断正确 / 实际评测 | 正确率 |
|---|---:|---:|
| single_call | 6 / 8 | 75.0% |
| solo | 8 / 8 | 100.0% |
| debate | 8 / 8 | 100.0% |

原始提交批次：`run_20261020T053935_586754Z_482b93ce9db749fd8fa61ee152e02673`、`run_20261020T054000_734977Z_a2eb78b5d1b14ef694bb3c6437055ff8`、`run_20261020T054105_239974Z_9ac24e2506f34a7a9c805b6d55207815`、`run_20261020T054111_920422Z_8bc916ae9f4c401288974e4fbd153a05`、`run_20261020T054516_484475Z_e0dbe200acb44da8ab2ce337ed0750b5`、`run_20261020T054524_008209Z_1a32dcc96eb74c90bce708ca2582b196`、`run_20261020T054528_362467Z_048eeda23d2b4cdab2c747bcdaeaff63`、`run_20261020T054532_798690Z_4ee6e66e05d146f5abd013111da957c1`。按 rN 核验共同 verifier 指纹；保留原始提交身份。

已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。

完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。

样本量仅 6 例。单次胜负不能证明普遍工具收益或 debate 优势。

<!-- END REAL KERNEL OCTOBER20 RESULTS -->

### Trace review

The first trial gives no tools 5/6 correct, Solo 6/6 and Debate 6/6. Only case 166 has a wrong no-tool decision with correct results from both tool workflows. The other five cases stay in the set as ordinary cases. Model and numerical parameters were not changed after these results.

**Case 166: a missed error term and final rounding.** The no-tool model uses `z^9 / 9` to estimate a remainder whose leading term is `2 * z^9 / 9`. It then asserts that FP32 rounding stays within the remaining tolerance. At `x = 3378`, the series and decimal coefficient errors total about `8.49e-7` before FP32 evaluation. The actual output error is `1.30e-6`. Solo [t8](../traces_glm/case_166/solo/r1/probes/t8_probe.py) and Debate [t9](../traces_glm/case_166/debate/r1/probes/t9_probe.py) both execute the actual kernel and find the same legal failure. Their causal explanations are less precise than this numerical evidence. Debate r2 later reproduces all 55 failures, but also overstates a seven-value large-input check as general coverage. Debate does not cover every legal value, despite its summary; one valid failure is sufficient for Reject.

**The exp2 pair does not separate the methods.** No tools evaluates the polynomial at `x = 0.5` and correctly rejects case 163. All three methods accept case 164. Tool runs reproduce the independent scalar results. Both Debate runs also contain an earlier probe with a missing scale factor or a wrong reference comparison. They discard those results before using actual-kernel checks.

**The correct log1p control does not separate the methods.** All three methods accept case 165. The no-tool explanation understates the maximum reduced argument. Debate initially claims to test a wide interval with a generator that produces only seven distinct values. Its final probe correctly checks all 47,104 legal scalar encodings. These errors do not change the final correct verdict, but they limit the claimed quality of the reasoning and coverage.

**The AES pair does not separate the methods.** No tools correctly checks the field representation in case 167. For case 168, it identifies the inconsistent extension constant and computes the valid counterexample `S(0x53) = 0x17`, while AES requires `0xed`. A more complex algebraic implementation was still solvable by reasoning alone.

**Some AES tool evidence is flawed.** Case 168 Solo repairs three failed probes before its valid final comparison finds 240/256 mismatches. Debate's main probe instead evaluates the candidate on S-box outputs and compares them to those same outputs. That comparison would reject a correct AES implementation too; its claimed 255/256 mismatch count is invalid. Separate probes t10 and t11 compare the candidate and reference on the same legal inputs and support Reject. Case 167 Solo's input-preservation assertion includes `or True`, so it cannot establish preservation. The independent qualification checks preservation correctly. These probe failures remain in the traces.

Full scalar enumeration is useful evidence, but it does not by itself prove all lengths, offsets, storage properties or GPU environments. The author qualification and the model's own evidence are reported separately. The results compare complete workflows and do not isolate tool access from extra model turns. They do not show that Debate is better than Solo.

### Fixed repeat check for case 166

The repeat rule was saved before any model call: at most the first two cases with an explicit no-tool error and valid correct tool results receive two fresh trials per method. Only case 166 qualified. The source, public requirements, model and budgets stayed unchanged.

| Trial | No tools | Solo + tools | Debate + tools |
|---|---|---|---|
| r1: initial trial | Accept, wrong | Reject, correct | Reject, correct |
| r2: fresh repeat | Accept, wrong | Reject, correct | Reject, correct |
| r3: fresh repeat | Reject, correct | Reject, correct | Reject, correct |

The no-tool result is correct in 1/3 trials overall, or 1/2 fresh repeats. The initial trial selected this case, so it must be shown separately from the fresh repeats. The r3 no-tool answer uses a rough combined-error estimate and does not produce a precise legal counterexample. It still counts as a correct verdict. This is an intermittent workflow difference, not a stable separation.

Both tool workflows are correct in 3/3 trials, or 2/2 fresh repeats, with legal executed counterexamples. Debate r2 includes one intermediate response that reached its 8,192-token turn cap; the same trial continued within its 32,768-token total cap and returned a valid verdict. The trace retains that response. No model trial was discarded or replaced.

The complete run contains 24 model trials and 41 probe attempts. All 24 trials have complete raw API captures and pass the source and configuration audit; this does not imply that every probe was valid. The recorded API estimate is **$3.675289**: $0.245587 for no tools, $1.073828 for Solo and $2.355874 for Debate. The estimate uses the saved pricing rates and excludes the Modal bill. Combining the selected repeats with the initial trials is an execution summary, not an unbiased six-case accuracy estimate.

The next construction should test the sum of approximation error and finite-precision reconstruction error under a fixed public tolerance. This round gives less support for increasing algebraic complexity alone: the no-tool model solved both the exp2 and AES pairs. Further repeats or new held-out implementations are needed before treating case 166 as stable. All sources, the pre-model protocol, qualification records and traces are retained.
