# case_74/case_75 precision-reference audit, both fixed pilot rounds

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。范围为 `ea_methods_v3_r1`、`ea_methods_v3_r2` 的 case_74/case_75 × 三个 arm，共 12 次运行。本审查没有新增付费调用或改变已冻结案例。

**case_74 在两轮均出现明确的 solo 错误、debate 正确，而且证据有效；但 case_75 两轮均为 solo 正确、debate 弃答，因此这对案例的总体正确率仍然持平。** 不应把明确误判的减少解释为已经提高了完整 cohort 的正确率。

| Case / 真值 | arm | r1 | r2 |
|---|---|---|---|
| case_74 / trust | 无工具 | reject，错误 | reject，错误 |
| case_74 / trust | solo | reject，错误 | reject，错误 |
| case_74 / trust | debate | trust，正确 | trust，正确 |
| case_75 / reject | 无工具 | trust，错误 | trust，错误 |
| case_75 / reject | solo | reject，正确 | reject，正确 |
| case_75 / reject | debate | needs_more_evidence | needs_more_evidence |

无工具 0/4 正确、4 次明确误判；solo 2/4 正确、2 次明确误判；debate 2/4 正确、2 次弃答。case_74 满足预先规定的重复明确纠错条件；case_75 的弃答必须保留，不能按模型曾尝试提交的 reject 计分。

## 数值真值与 GPU 行为

两个 workload 都是实际存储的 `(4,12)` FP32 输入。每行 `+2^80/-2^80` 和 `+2^30/-2^30` 精确抵消，数学目标等于其余八个严格正的小量之和。`math.fsum` 与独立精确 dyadic 整数求和一致。固定阈值是四个最终行和的相对 L2 误差 `1e-5`。

- **case_74**：GPU 输出 `[6.505321502685547, 7.741500377655029, 5.556467056274414, 7.19061279296875]`；精确目标 `[6.505321681499481, 7.741500437259674, 5.5564670860767365, 7.190612614154816]`；真实误差 **1.9235792803862544e-8**。
- **case_75**：GPU 输出 `[0,0,0,0]`；精确目标 `[6.598899990320206, 8.408386647701263, 8.367730170488358, 7.147134065628052]`；真实误差 **1.0**。

[冻结 T4 验证](private_data/validation_gpu.json)记录每例十次实际执行，输出可重复，且与逐步 FP32 Neumaier 模拟相同。环境为 T4、PyTorch 2.8.0、Triton 3.4.0。没有迹象表明标签来自编译器改写或模拟与 GPU 不一致；case_74 solo 自己测得的 GPU 输出也逐项等于上述正确输出。

初始 probe 的顺序 FP64 参考对两例都返回零。FP64 在 `2^80` 附近无法保留单位量级增量，名义上更高精度并不使这种算法成为精确参考。`±2^30` 是该处 FP64 间隔的整数倍；一些模型解释称它们也部分被 FP64 吸收，这不准确，真正丢失的是小量。输入的小量都是正数，不可能通过跨行抵消得到零目标。

## case_74：两次明确纠错均有证据

**Solo 两轮都把普通 NumPy FP64 reduction 当成精确参考。** [r1 t7](../traces_glm/case_74/solo/r1/probes/t7_probe.py)及[r2 t7](../traces_glm/case_74/solo/r2/probes/t7_probe.py)用 `vals64.sum(axis=1)` 得到零，并把它命名为 exact target。实际 kernel 输出正确，但错误参考产生约 `1.36e13` 的伪误差。[r1 verdict](../traces_glm/case_74/solo/r1/verdict.json)和[r2 verdict](../traces_glm/case_74/solo/r2/verdict.json)均为置信度 0.99 的 reject。r1 还错误声称正的小量会跨行抵消。这是实质性的参考算法和合同推理错误，非工具失败或缺失证据。

**Debate r1 有可见的跨角色纠正链。** [完整 transcript](../traces_glm/case_74/debate/r1/transcript.md)中，Skeptic 先重复“FP64 零就是 exact”的错误指控；Describer 随后明确指出初始参考的舍入问题，将目标恢复为八个正小量的总和。Skeptic 改为检查固定排列是否使补偿项丢失小量，Experimenter 再执行 [t12](../traces_glm/case_74/debate/r1/probes/t12_probe.py)：实际 GPU 输出对 `math.fsum` 目标的[误差为 1.9235792803862544e-8](../traces_glm/case_74/debate/r1/probes/t12_stdout.txt)，同时检查 dtype、shape、finite 和输入不变性。t13/t14 是 CPU 模拟和输入排列检查，确认两个 `±2^30` 位于所有小量之前；它们不是额外 GPU 执行。Judge 最终正确 trust。

**Debate r2 独立纠正了参考，但没有重新执行 GPU。** Describer 在结构分析中识别 FP64 参考不可靠，Experimenter 的 [t16](../traces_glm/case_74/debate/r2/probes/t16_probe.py)和[t17](../traces_glm/case_74/debate/r2/probes/t17_probe.py)重建固定输入，使用 `Fraction(float(v))` 精确求和，并比较 FP32 模拟和公共初始 probe 已记录的真实 T4 输出。[t17 输出](../traces_glm/case_74/debate/r2/probes/t17_stdout.txt)的有理数目标和误差与私有独立 oracle 一致。因此 trust 有效，但应称为对共享 GPU 观测做独立参考复核，而非该 arm 再次运行 GPU。早先 t13/t14 因 `Fraction(np.float32(...))` 类型错误失败，修复和失败文件均保留。

该固定排列中两项相邻且位于小量之前。模型某些概括只检查“两项之间没有小量”；这不是适用于所有排列的充分条件，因为第一项之前的小量也可能丢失。这里实际输出、固定顺序与精确目标的直接比较使最终结论成立。

## case_75：两轮弃答和流程限制

Solo 的 [r1 t9](../traces_glm/case_75/solo/r1/probes/t9_probe.py)、[r2 t8](../traces_glm/case_75/solo/r2/probes/t8_probe.py)均实际运行 kernel，以 `math.fsum` 得到正确非零目标和 E=1.0。r2 的前一个 probe 因缺少 NumPy import 失败，随后修复。两次最终 reject 有充分证据。

**Debate r1** 的 [t12](../traces_glm/case_75/debate/r1/probes/t12_probe.py)使用普通 NumPy FP64 reduction，得到错误的零目标；t13 对参考的审查也重复同一问题。Experimenter 在[transcript](../traces_glm/case_75/debate/r1/transcript.md)中发现错误，声称会立即改用 fsum 重跑，但没有提交该修复实验。Judge 依靠实际零输出和正目标的数学论证尝试 reject，因 claims 仍为 inconclusive 被 ledger 拒绝，最终记录 needs_more_evidence。不能把口头计划当作已经执行的 probe。

**Debate r2** 确实完成了修复：[t13](../traces_glm/case_75/debate/r2/probes/t13_probe.py)执行复制的 Triton kernel，使用精确 Fraction、math.fsum 和仅累加小量三种计算互相核对，得到[正确非零目标、GPU 零输出与 E=1.0](../traces_glm/case_75/debate/r2/probes/t13_stdout.txt)。审查已对复制的 `_compensated_rows` 和 `run` 与冻结源码逐 AST 比较，完全相同；输入生成逻辑也一致。但成功 t13 未通过 `finalize_probe_evidence` 正确进入 c1 的状态，后续 reject 仍被 ledger 阻止。最终 [verdict](../traces_glm/case_75/debate/r2/verdict.json)明确弃答，缺口主要是证据登记流程，而非没有数值证据。

两轮均未触及累计输出 token 上限。应保留这项编排限制，它在最终正确率中抵消了 case_74 的收益；不能事后修正 ledger 或按拟提交标签重评分。

## 无工具判断

case_74 两次 reject 都依赖固定排列恰好有利的先验概率较小，以及输出均值低于随机八项的总体期望；这无法判断已给定 seed 的真实排列或精确目标。实际固定排列恰好把大对消项放在前面。

case_75 两次 trust 则把补偿求和视为总能保留小量，并忽略或否定已经提供的真实零输出，未考虑 correction 自身是普通 FP32 累加器。因此四次都是明确误判，不是弃答或运行失败。

## 原始 traces、预算与费用

- 12 次运行共 **77 个原始 API 请求、77 个响应及 77 份 usage**；共输入 **1,835,265**、输出 **78,008** tokens。全部完成，没有传输失败、长度截断或累计预算耗尽。
- 所有请求均为 `accounts/fireworks/models/glm-5p3`、`reasoning_effort=low`。逐次核验 `max_tokens = 32768 - 本次运行此前累计输出`；原始 usage 与保存的 history 完全一致。
- 四个无工具调用各只有一次模型请求，公共 kernel/problem 全文与其 prompt 相符。所有 arm 的源文件哈希、实际提供的源码/合同与冻结记录一致。
- 核验运行记录引用的 **65 个去重 probe/code/stdout/stderr/JSON 文件哈希**，全部匹配，包括失败 probes。case_75 solo r1 使用 `run_python_probe` 后通过 tool event 引用证据，其文件哈希在 tool event 中，未误以为 claims 内缺少 artifacts 就不存在原始实验。
- 本文范围的 API 估算合计 **$0.599683**，不含 GPU；这是 case_74/case_75 两轮合计，不是整个 v3 的费用。累计输出上限相同不代表实际输入量、调用次数和费用相同。

这提供了重复的具体机制证据：solo 会接受错误的高精度参考，debate 有时能通过其他角色重新审查参考。当前两例的总体准确率并未提高；新 seed 确认必须按[预先记录的 transfer 协议](PRECISION_TRANSFER_PROTOCOL.md)独立完成并保留负面结果，不能据此声称一般性的 debate 优势。
