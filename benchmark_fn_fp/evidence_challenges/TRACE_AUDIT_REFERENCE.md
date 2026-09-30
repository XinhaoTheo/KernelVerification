# case_62/case_63 pilot trace audit

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。范围：`ea_pilot_r1` 的 case_62/case_63 × solo/debate，共四次带工具运行。
审查仅读取已保存请求、响应、probe、结果和冻结真值，没有追加付费实验。

**四次最终判断均正确，且均有符合合同的独立 FP64 主实验支持。Solo 一次实验就识别并解决了初始测试的共同误差，未观察到 debate 的额外判断优势。**

| 案例 / arm | 判断 | 独立主实验相对误差 | 执行 probes | API 调用 | 输入 / 输出 tokens | API 估算 |
|---|---|---:|---:|---:|---:|---:|
| case_62 solo | trust，正确 | 0.00206523275535747 | 1 | 5 | 66,848 / 1,508 | $0.020376 |
| case_62 debate | trust，正确 | 0.00206523275535756 | 2 | 9 | 195,030 / 6,464 | $0.061719 |
| case_63 solo | reject，正确 | 0.255847958607997 | 1 | 5 | 66,660 / 1,757 | $0.020598 |
| case_63 debate | reject，正确 | 0.255847958607997 | 1 | 9 | 192,906 / 4,946 | $0.059454 |

合同阈值为 0.05，误差分母均正确采用 `max(norm(prediction), 0.1)`。
费用来自项目现有 GLM Fireworks 价格配置和原始 usage，不含 GPU。
这四次合计 $0.162147；solo 两次 $0.040974，debate 两次 $0.121173。

## 决定性证据

- **case_62 solo**：[t7 代码](../traces_glm/case_62/solo/r1/probes/t7_probe.py)、[实际结果](../traces_glm/case_62/solo/r1/probes/t7_stdout.txt)、[最终判断](../traces_glm/case_62/solo/r1/verdict.json)。从实际 FP32 输入转为 FP64，居中形成 ridge normal equations，`numpy.linalg.solve` 后比较四个最终预测。未采用初始 probe 的 FP32 参考作为真值。
- **case_63 solo**：[t7 代码](../traces_glm/case_63/solo/r1/probes/t7_probe.py)、[实际结果](../traces_glm/case_63/solo/r1/probes/t7_stdout.txt)、[最终判断](../traces_glm/case_63/solo/r1/verdict.json)。同样独立居中计算，找到 25.5848% 的预测误差；初始 probe 的零误差被明确识别为共同使用 FP32 raw moments 所致。
- **case_62 debate**：[t12 代码](../traces_glm/case_62/debate/r1/probes/t12_probe.py)、[实际结果](../traces_glm/case_62/debate/r1/probes/t12_stdout.txt)、[最终判断](../traces_glm/case_62/debate/r1/verdict.json)。主实验使用 CUDA PyTorch FP64 居中求解，最终输出与参考、误差均匹配冻结真值。额外 t13 的解释存在下述问题，但 t12 独立支持 trust。
- **case_63 debate**：[t10 代码](../traces_glm/case_63/debate/r1/probes/t10_probe.py)、[实际结果](../traces_glm/case_63/debate/r1/probes/t10_stdout.txt)、[最终判断](../traces_glm/case_63/debate/r1/verdict.json)。主实验与 solo 相同，给出正确 reject；辅助协方差归因存在下述偏差。

所有主实验的 GPU 输出逐项匹配[冻结 T4 真值](private_data/validation_gpu.json)，参考预测与私有独立参考的差异均在 FP64 舍入量级。实验都使用规定种子；没有引入范围外样例，也没有修改候选源码。

## 额外证据中的问题

**case_62 debate t13 没有准确复现 initial_probe。** [t13 代码](../traces_glm/case_62/debate/r1/probes/t13_probe.py)在 FP32 累加后用 `[float(v) for v in stats/32]` 转为 Python float，随后协方差、求解和查询预测采用更高精度。它报告的“initial_probe reference”实际是混合精度变体：[相对误差 0.00311503097](../traces_glm/case_62/debate/r1/probes/t13_stdout.txt)。真实初始参考与候选输出相同，相对于合同参考的误差是 0.00206523276。两者均低于阈值，最终标签不受影响；“exactly reproduce”以及 Judge 引用的 0.00312 不能视为真实初始参考的测量。

此外，[c2](../traces_glm/case_62/debate/r1/claims.json)的原命题是“初始零误差不能建立合同满足”，这一证据不足判断仍成立。新独立实验确认内核合格，不能反过来证明原始共同误差实验本身充分。Debate 将 c2 标为 rebutted，在命题层面不准确。Skeptic 的末轮复核未纠正该问题。

**两份 debate 辅助 raw-moment 诊断不完全复制候选的归约顺序。** case_62 t12 使用向量 `.sum()`，候选是逐元素 FP32 累加；它的协方差/行列式诊断只能描述该向量归约变体。case_63 t10 使用 `np.mean` 求特征均值，候选则逐元素累加。case_63 的辅助结果为 covariance `0.005859375`、相对误差 2.23705%；按公开候选顺序做的 CPU FP32 仿真为 `0.0057373046875`、相对误差约 0.107112%。因此 Judge 将 2.24% 直接称为 kernel 的该协方差误差缺乏依据。这里没有额外执行 GPU 插桩；CPU 仿真数值用于说明归约顺序差异。决定性的真实 GPU 最终预测误差 25.5848% 仍然有效。

四次运行均遇到可恢复的 ledger 参数遗漏；case_62 debate 还曾提交格式不合规的 description update。所有错误在原 trace 中保留，后续成功修复；五个在 GPU 环境执行的 probes 均成功，无编译错误或超时。其中四个主实验实际执行候选 kernel；t13 仅检查参考算法，没有调用候选 kernel。

## Trace、冻结和预算核验

- 四次运行共 **28 个 API 请求、28 个响应、28 份完整 usage**，数量与保存的模型 history 一致；没有 `finish_reason=length`。
- 所有实际请求均使用 `accounts/fireworks/models/glm-5p3`、`reasoning_effort=low`。逐调用核对 `max_tokens <= 32768 - 此前实际累计输出`，四次累计输出均未超过 32,768；这是累计输出预算匹配，不代表输入 token、费用或 GPU 时间相同。
- `trace_meta.json` 的 kernel/problem 哈希、`run.json` 内实际提供的源码/合同及当前公开文件均与冻结 GPU 记录一致。公开合同含真实初始 probe 输出，未含私有标签。
- 核验 claims 所引用的 **20 个 probe 代码/stdout/stderr/JSON 文件哈希**，全部匹配。四个主实验的参考使用合同定义的原始输入和 FP64 居中计算；额外 t13 按上述限制解释。

该 pilot 对共同误差测试审查给出的结果是：两种工具系统都能纠正误导性通过，solo 已足够；debate 的额外讨论还产生了未被复核纠正的辅助测量和命题解释问题。该结论仅覆盖这对案例的一次预先固定运行。
