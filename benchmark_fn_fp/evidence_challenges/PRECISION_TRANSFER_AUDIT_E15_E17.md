# Precision transfer: case_76–case_78 trace audit

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。范围为 `ea_precision_transfer_r1` 的 case_76/case_77/case_78 × single_call/solo/debate 九次运行。审查没有新增模型或 GPU 调用，没有修改公开案例、冻结真值和原始 traces。

**这一半 cohort 中，无工具 1/3、solo 2/3、debate 3/3。case_78 是有效的单次新种子差异：solo 误把普通 FP64 求和当成精确参考而拒绝，debate 用 Fraction 真值与实际 GPU 输出比较后正确接受。case_76/case_77 的 solo 也主动使用 Fraction，不能把这次差异描述成 solo 普遍不会审查参考。**

| 案例 / arm | 最终标签 | API 请求 | 输入 / 输出 tokens | API 估算 |
|---|---|---:|---:|---:|
| case_76 single_call | reject，错误 | 1 | 1,358 / 4,236 | $0.005040 |
| case_76 solo | trust，正确 | 6 | 77,332 / 2,696 | $0.024619 |
| case_76 debate | trust，正确 | 9 | 202,445 / 10,046 | $0.067735 |
| case_77 single_call | trust，错误 | 1 | 1,363 / 2,168 | $0.002766 |
| case_77 solo | reject，正确 | 5 | 60,157 / 2,329 | $0.019406 |
| case_77 debate | reject，正确 | 10 | 314,608 / 14,877 | $0.104455 |
| case_78 single_call | trust，正确 | 1 | 1,357 / 2,048 | $0.002633 |
| case_78 solo | reject，错误 | 5 | 58,289 / 1,403 | $0.017864 |
| case_78 debate | trust，正确 | 9 | 237,517 / 10,009 | $0.077515 |

此范围合计 47 个请求、954,426 输入 tokens、49,812 输出 tokens，API 估算 $0.322033；分组为无工具 $0.010439、solo $0.061889、debate $0.249705。不含 GPU。共同输出上限不等于输入 token、费用或 GPU 时间相同。

## case_78 的有效差异来自哪里

[Solo t7 代码](../traces_glm/case_78/solo/r1/probes/t7_probe.py)正确运行了真实 kernel，却用 `vals.sum(axis=1)` 对 FP64 数组求和作为参考，并注释“exact enough”。[实际结果](../traces_glm/case_78/solo/r1/probes/t7_stdout.txt)中参考为 `[0,0,0,0]`，GPU 输出约 `[7.9236,7.55135,7.47421,8.69661]`，于是报告相对误差 `1.5852478567108e13`。[最终 reject](../traces_glm/case_78/solo/r1/verdict.json)直接称全零结果是 exact target，confidence=0.99。这不是执行失败、弃答或预算耗尽，而是明确参考错误。代码提到“also use Kahan on float64”，但没有实现该额外检查。

合同要求对实际储存的 FP32 输入做精确实数求和；两对大数抵消后八个正小量仍应保留。独立 [oracle 审查](PRECISION_TRANSFER_ORACLE_AUDIT.md)已用 Fraction 对全部 512 个候选验证真值，case_78 参考为：

`[7.923600226640701, 7.551349997520447, 7.474209100008011, 8.696605235338211]`。

[Debate t13 代码](../traces_glm/case_78/debate/r1/probes/t13_probe.py)使用 `Fraction(float(v))` 求实际 FP32 数值的精确和，并调用公开原 kernel 的 `make_inputs()` / `run()`。[输出](../traces_glm/case_78/debate/r1/probes/t13_stdout.txt)的 GPU 结果与参考逐值匹配冻结记录，相对 L2 误差 **3.3787355339815074e-8 < 1e-5**，因此 [trust](../traces_glm/case_78/debate/r1/verdict.json)有独立决定性支持。该 probe 没有另做 dtype/immutability 检查；它引用共同初始 probe 的结构检查，且公开源码与独立冻结验证支持结构要求。没有把新的未测量结构字段伪造为 probe 输出。

[角色 history](../traces_glm/case_78/debate/r1/run.json)显示，Describer 一开始就识别全零 FP64 参考有问题；Skeptic 同时要求检查 correction 累加中的丢失，以及输出对精确真值的误差；Experimenter 实施 Fraction＋GPU 测量，Judge 据此接受。这里有实际多角色证据链，但**不是某个角色先给错误 verdict、另一个角色再纠正它的 trace**，也没有单独排除更多提示或更多计算量的作用。

辅助 [t12](../traces_glm/case_78/debate/r1/probes/t12_probe.py)正确发现大数在第 1、2 列先相邻抵消，其简化 correction 仿真也匹配真实输出。但其另一个“full kernel reduction”把 total 初始化为 `2^80`，随后又遍历含 `2^80` 的首项，重复计入首项，导致 [sim_kernel_out](../traces_glm/case_78/debate/r1/probes/t12_stdout.txt)四行均为 `2^80`。原 trace 没有解释或修复这一分歧。不能概括为所有模拟都正确；有效的是简化 correction 模型和独立 t13 精确测量。该辅助 bug 不改变 case_78 的正确 verdict。

## case_76/case_77 的主证据

**case_76 solo**：[t8](../traces_glm/case_76/solo/r1/probes/t8_probe.py)直接运行原 kernel，用 Fraction 真值比较，[误差](../traces_glm/case_76/solo/r1/probes/t8_stdout.txt)为 `7.975729244084926e-8`。输入前后比较有效。case_76 debate [t12](../traces_glm/case_76/debate/r1/probes/t12_probe.py)运行原 kernel，用 `math.fsum` 求真值，[结果](../traces_glm/case_76/debate/r1/probes/t12_stdout.txt)相同；`math.fsum` 在本数据上的结果已被独立 Fraction 全量验证。其 [t13](../traces_glm/case_76/debate/r1/probes/t13_probe.py)仅在 CPU 复现初始顺序 FP64 的零参考。两组主实验的输出/参考均逐值匹配[冻结数据](private_data/validation_gpu.json)。

case_76 的 passing 机制是 `-2^30,+2^30` 恰好占据最前两个 interior 列，八个小量全部在其后；只说“负大数在正大数之前”不足以保证不丢失。Debate 的简略最终解释遗漏了相邻且先于小量的条件，但 t12 打印的实际顺序明确满足。其 t13 正确打印 FP64 在 `2^80` 向上间距为 `2^28`，Judge 仍写约 `2^27`；此外 `2^30` 项没有被 FP64 吞掉，它们可以精确加减，丢掉的是约 1 的小项。以上是辅助解释精度问题，不影响正确参考测量。

**case_77 solo**：[t7](../traces_glm/case_77/solo/r1/probes/t7_probe.py)运行原 kernel，以 `Fraction(v.item())` 形成真实参考，[相对误差](../traces_glm/case_77/solo/r1/probes/t7_stdout.txt)为 `0.8856309517807359`，而非对错误零参考得到的巨大数值。dtype/shape/finite 和输入不变性都有实际检查。最终 reject 正确；但其归因把丢失过度归结于 `2^80` 主累加器，实际关键是 correction 本身遇到 interior `±2^30` 后丢失已有/后来的小项。

case_77 debate [t20](../traces_glm/case_77/debate/r1/probes/t20_probe.py)把候选实现复制到临时模块后编译运行，而没有直接 import 原公开文件。本审查逐 AST 核对 `_compensated_rows`、`run`、`make_inputs_numpy`、`make_inputs` 四个函数，与公开源完全一致，seed、num_warps、关闭 fusion 的设置也一致；[GPU 输出和 Fraction 参考](../traces_glm/case_77/debate/r1/probes/t20_stdout.txt)逐值匹配冻结记录，因此该实验仍有效。它没有测试一个被修复或简化的 kernel。CPU [t19](../traces_glm/case_77/debate/r1/probes/t19_stdout.txt)独立确认非零参考，[t21](../traces_glm/case_77/debate/r1/probes/t21_stdout.txt)逐步记录 correction 被 `-2^30` 清除已有小和、随后吞掉小项、最终只保留最后一个小量；这与失败机制一致。

## 无工具结果与失败记录

[case_76 source](../traces_glm/case_76/single_call/r1/verdict.json)描述了一个可能发生的 correction 丢失，但没有计算固定排列，因而误拒。它给出的“仅需大数相邻”也忽略之前小量仍可能在遇到大数时丢失。[case_77 source](../traces_glm/case_77/single_call/r1/verdict.json)发现初始零参考错误后，直接推断补偿输出匹配真值，误接收；实际上八个小量各至少 0.25，精确和至少为 2，而输出四行都小于 1.35。[case_78 source](../traces_glm/case_78/single_call/r1/verdict.json)标签正确。评分按固定二元标签，不把缺少完整证明自动改判为错误。

case_76 solo 的 t7、case_77 debate 的 t15/t16/t17 都因 `Fraction(np.float32)` 不被 Python Fraction 构造器直接接受而失败；后续通过 `float(v)` 转换修复。这一转换精确保存 FP32 数值，不是近似十进制参考。原始失败代码、stderr 和工具事件全部保留。前两例中执行 GPU 后才在 Fraction 参考计算失败的尝试不算成功数值证据。没有隐藏模型重试。

## 原始 traces、预算和范围

九次运行的 **47 个 request、47 个 response、47 份 usage**完整，工具组 raw 调用数与 history usage 数一致。每个实际请求均为 GLM-5p3、low reasoning，且 `max_tokens == 32768 - 此前累计实际输出 tokens`；每个 run 未超累计上限，没有 length 截断、provider 错误或弃答。三个单次调用各恰好一个无工具请求。

公开源码/附带真实初始观察的合同、trace metadata 哈希、工具 run 实际提供的材料和 source-only prompt 全部匹配冻结版本。15 个 probe 执行中 11 个成功、4 个参考构造报错；8 次包含真实候选 GPU 执行，其中两次在 GPU 执行后参考构造报错，另有 7 次为 CPU 分析。case_77 debate 的两次 GPU 执行使用已核验等价的源码副本。工具事件引用的 **56 个文件哈希**全部匹配，失败文件也完整。

本范围只有一个新种子上的 solo-wrong/debate-correct。是否启动整批 r2 必须依据完整 case_76–case_81 cohort 的预定规则，不能用这三例的局部结果替代全批差值门槛；本报告不宣称已经取得重复的新种子优势。
