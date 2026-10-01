# 无工具单次调用 vs 带工具验证：case_38–case_61

这 24 题、12 对机制判断的是：**公开的固定输入经过近似计算后，最终误差是否满足合同**。它们主要区分源码估测与实际计算；已完成的指定复验中，solo 和 debate 的最终标签准确率均打平，不能据此声称多角色比单 agent 更强。

公开 kernel、输入生成器和合同统一位于 [triton_eval_cases](../triton_eval_cases/)。本目录只保留这份说明和 `private_data/`；构造、独立参考、GPU 验证及报告程序在 [eval_scripts/single_call_vs_tools_challenges](../eval_scripts/single_call_vs_tools_challenges/)。新 CLI 名称是 `single_call_vs_tools_challenges`，历史 trace 的逻辑 dataset `numerical_challenges` 保留用于追溯。

## 题型与真值

每对使用同一计算和合同，仅公开 seed 或排列不同；包含一个合格和一个不合格输入。下面误差均为合同原始比例，**不是百分数**；精确值、输入/源码/输出 hash 见 [冻结 T4 验证](private_data/validation_gpu.json)。

| 案例与合同 | 需要验证的量 | T4 误差，按案例顺序 | 容差 |
|---|---|---:|---:|
| [38/39](../triton_eval_cases/case_38/problem.txt) | logit 量化经过 softmax 后，与 value 排列共同决定的 attention 误差 | 0.00809816 / 0.03652389 | 0.02 |
| [40/41](../triton_eval_cases/case_40/problem.txt) | 64 步、16 维非正规递推中的 FP16 状态舍入 | 0.000769029 / 0.004907839 | 0.002 |
| [42/43](../triton_eval_cases/case_42/problem.txt) | 相同项按不同顺序做 FP32 顺序求和 | 0.90019872 / 0 | 0.1 |
| [44/45](../triton_eval_cases/case_44/problem.txt) | LayerNorm 原始矩方差 E[x²]−E[x]² 的相消 | 0.000242616 / 0.144819646 | 0.02 |
| [46/47](../triton_eval_cases/case_46/problem.txt) | 16×16 SPD 系统、64 次 Richardson 迭代后的解误差 | 0.02363685 / 0.12440400 | 0.08 |
| [48/49](../triton_eval_cases/case_48/problem.txt) | 禁用 FMA 的 48 阶 FP32 Horner 求值 | 0.0000337287 / 0.000472925 | 0.0002 |
| [50/51](../triton_eval_cases/case_50/problem.txt) | 近共线正交投影的小残差归一化 | 0.001543649 / 0.060455265 | 0.01 |
| [52/53](../triton_eval_cases/case_52/problem.txt) | 1/8 网格坐标量化改变最近邻路由后的 embedding | 0 / 1.34018400 | 0.1 |
| [54/55](../triton_eval_cases/case_54/problem.txt) | 32 点中点积分对连续振荡积分的近似 | 0.001552520 / 0.118884368 | 0.035 |
| [56/57](../triton_eval_cases/case_56/problem.txt) | 16 个 Fourier 模式仅保留前 6 个后的完整输出 | 0.071709356 / 0.230464266 | 0.15 |
| [58/59](../triton_eval_cases/case_58/problem.txt) | 病态 SPD 矩阵 FP32 消元的 log-determinant | 0.000000202355 / 0.000711920 | 0.0001 |
| [60/61](../triton_eval_cases/case_60/problem.txt) | 平方距离展开式相消传播到归一化 RBF 预测 | 0.000142222 / 0.292406361 | 0.05 |

只有 42/43 的标签顺序是 reject/trust；其余均为 trust/reject。合同只涵盖声明的输入，不能另造输入来推翻合格案例。风险上界或中间量误差也不能直接替代最终输出指标。

## 实验设置与结果

共同模型为 Fireworks `accounts/fireworks/models/glm-5p3`，三臂是无工具单次调用、solo＋GPU 工具、describer/skeptic/experimenter/judge 四角色 debate＋GPU 工具。Solo 最多十轮、debate 四轮；本组的 token cap 是**每次调用**上限，没有匹配总调用、总 tokens、费用或 GPU 时间。所有 arm 得到相同公开材料，容器不挂载私有参考、搜索日志或答案。

原始批次名保存在 `trace_meta.json.original_trial`；当前 `rN` 是每个 case/arm 内的存储序号，不能把相同序号当作相同实验设置。错误、弃答、截断、服务错误和未完成分别统计。失败槽位不会用后来的成功替换，API 估算不含 Modal 或未返回 usage 的潜在费用。

### 初始六题：case_38–case_43

2026-09-23 原协议设置每调用 65536 tokens，先覆盖六题三臂，再检查最早三次 source 与各最早两次工具尝试。目标是至少两个机制各有 source 明确误判 ≥2/3，且两工具臂各 2/2 正确；早期失败保留，不能被后来的 low 配置复验追溯替换。

早期 `numerical_r3/r4`、`numerical_probe32768`、`numerical_smoke8k` 等包含 504、超时、截断和中断预留记录。之后单列 `numerical_low_r1/r2`：low reasoning、每调用 8192 tokens，source 分别 4/6、2/6，solo/debate 每轮均 6/6。Source 第一轮错 case_41/43，第二轮错 case_38/39/41/43；两轮费用分别 $0.387847、$0.410533。这说明该 low 配置下工具收益可重复，并不意味着原始 65K 协议的门槛已满足，也不能仅凭先后顺序确定早期失败的全部原因。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED MAIN RESULTS -->

生成时间：2026-10-01T04:01:49.570205+00:00

初始组 case_38–case_43；后续两组采用各自约定的实验协议。协议与固定复验窗口见本页说明。

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| numerical_low_probe | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 1 | 0 | 1 | 0 | 0 | 0 | $0.001688 |
| numerical_low_r1 | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 10 | solo | 6 | 6 | 0 | 0 | 0 | 0 | $0.085919 |
| numerical_low_r1 | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 4 | debate | 6 | 6 | 0 | 0 | 0 | 0 | $0.286048 |
| numerical_low_r1 | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 6 | 4 | 2 | 0 | 0 | 0 | $0.015880 |
| numerical_low_r2 | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 10 | solo | 6 | 6 | 0 | 0 | 0 | 0 | $0.098141 |
| numerical_low_r2 | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 4 | debate | 6 | 6 | 0 | 0 | 0 | 0 | $0.300138 |
| numerical_low_r2 | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 6 | 2 | 4 | 0 | 0 | 0 | $0.012254 |
| numerical_probe32768 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 1 | 0 | 1 | 0 | 0 | 0 | $0.015838 |
| numerical_r1 | glm-5p3 / fireworks | 每次 65536 / 总 unknown / 推理 unknown / 轮次 10 | solo | 6 | 1 | 0 | 0 | 0 | 5 | $0.032374（不完整） |
| numerical_r1 | glm-5p3 / fireworks | 每次 65536 / 总 unknown / 推理 unknown / 轮次 4 | debate | 6 | 0 | 0 | 0 | 0 | 6 | $0.000000（不完整） |
| numerical_r1 | glm-5p3 / fireworks | 每次 65536 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 6 | 1 | 0 | 0 | 0 | 5 | $0.040983（不完整） |
| numerical_r3 | glm-5p3 / fireworks | 每次 65536 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 6 | 1 | 2 | 0 | 0 | 3 | $0.096281（不完整） |
| numerical_r4 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 6 | 1 | 1 | 0 | 4 | 0 | $0.192788 |
| numerical_smoke8k | glm-5p3 / fireworks | 每次 8192 / 总 unknown / 推理 unknown / 轮次 unknown | single_call | 1 | 0 | 0 | 0 | 1 | 0 | $0.009372 |

共 69 次记录；API 费用估算 $1.187704。费用未知 19 次，费用记录不完整 8 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

固定窗口复验门槛： **未达到**; 达到门槛的机制： none (0/2).

配对工具比较： 18; debate 纠正 solo 明确错误： 0; solo 纠正 debate 明确错误： 0. 这些选定案例的比较不能推出普遍优势。

<!-- END GENERATED MAIN RESULTS -->

</details>

### 扩展六题：case_44–case_49

在首次模型调用前固定 `extension_low32_r1/r2` 为六题三臂，`extension_low32_r3` 仅 source；均 low reasoning、每调用 32768。另对全部六题各做一次 `extension_default64_r1`，省略 reasoning_effort、上限 65536，作为独立配置对照。共 48 个指定槽位，不因初始结果选择重测对象。

Low 条件下 source 11/18（逐轮 3/6、3/6、5/6），两工具臂各 12/12；42 次均提交最终二元 verdict。Case_44/48 满足 ≥2/3 source 明确误判、两工具臂各 2/2 的预定两机制门槛。Default 对照中 case_44/46 错、45/47 对、48/49 为 504；不能把服务失败计成误判或默认配置下的稳定差距。全部记录估算 $1.055612，另有两次未返回 usage 的未知费用。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED EXTENSION RESULTS -->

生成时间：2026-10-01T04:01:49.973626+00:00；题目范围：case_44–case_49。

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| extension_default64_r1 | glm-5p3 / fireworks | 每次 65536 / 总 unknown / 推理 default / 轮次 unknown | single_call | 6 | 2 | 2 | 0 | 0 | 2 | $0.142321（不完整） |
| extension_low32_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 10 | solo | 6 | 6 | 0 | 0 | 0 | 0 | $0.090318 |
| extension_low32_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 4 | debate | 6 | 6 | 0 | 0 | 0 | 0 | $0.357550 |
| extension_low32_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 6 | 3 | 3 | 0 | 0 | 0 | $0.014530 |
| extension_low32_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 10 | solo | 6 | 6 | 0 | 0 | 0 | 0 | $0.098714 |
| extension_low32_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 4 | debate | 6 | 6 | 0 | 0 | 0 | 0 | $0.320481 |
| extension_low32_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 6 | 3 | 3 | 0 | 0 | 0 | $0.016824 |
| extension_low32_r3 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 6 | 5 | 1 | 0 | 0 | 0 | $0.014874 |

共 48 次记录；API 费用估算 $1.055612。费用未知 2 次，费用记录不完整 2 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

指定窗口复验门槛： **通过**; 达到门槛的题目： case_44, case_48.

缺失实验保持缺失，后续重试不替换原定记录。只有 CPU 标签不能满足 GPU 校验门槛。完整计划、逐题审核和配对比较可通过可选 JSON 导出查看。

<!-- END GENERATED EXTENSION RESULTS -->

</details>

### 后续十二题：case_50–case_61

首次调用前固定 `oz_low32_r1/r2` 为十二题三臂、`oz_low32_r3` 仅 source；另做一次全十二题 default/65536 source 对照。共 84 个 low 槽位＋12 个 default 槽位。目标为至少三个机制满足同样的 source ≥2/3 明确错误、工具各 2/2 正确；失败、弃答或预算耗尽不算明确错误，后续重试不能替换指定槽位。

Low 下 source 13/36 正确、19 次明确错误（9 次误拒、10 次误接受）、4 次弃答；弃答均在 case_56/57。Solo/debate 各 24/24。Case_50/51/52/55/59/61 达标，覆盖五个机制。Default 对照 4 对、2 错（52/61）、6 次 504（50/54/55/57/58/59），原失败均保留且未补跑。全批估算 $2.054085，未知失败费用和 GPU 另计；384 个 API 调用记录中 378 个有响应及 usage，六个保留部分失败 trace。90 个有响应的运行完整捕获原始 API 数据。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED OZ RESULTS -->

生成时间：2026-10-01T04:01:50.520862+00:00；题目范围：case_50–case_61。

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| oz_default64_r1 | glm-5p3 / fireworks | 每次 65536 / 总 unknown / 推理 default / 轮次 unknown | single_call | 12 | 4 | 2 | 0 | 0 | 6 | $0.167357（不完整） |
| oz_low32_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 10 | solo | 12 | 12 | 0 | 0 | 0 | 0 | $0.206290 |
| oz_low32_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 4 | debate | 12 | 12 | 0 | 0 | 0 | 0 | $0.691501 |
| oz_low32_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 12 | 3 | 8 | 1 | 0 | 0 | $0.027576 |
| oz_low32_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 10 | solo | 12 | 12 | 0 | 0 | 0 | 0 | $0.189598 |
| oz_low32_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 4 | debate | 12 | 12 | 0 | 0 | 0 | 0 | $0.721764 |
| oz_low32_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 12 | 6 | 5 | 1 | 0 | 0 | $0.023800 |
| oz_low32_r3 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 12 | 4 | 6 | 2 | 0 | 0 | $0.026199 |

共 96 次记录；API 费用估算 $2.054085。费用未知 6 次，费用记录不完整 6 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

指定窗口复验门槛： **通过**; 达到门槛的题目： case_50, case_51, case_52, case_55, case_59, case_61.

缺失实验保持缺失，后续重试不替换原定记录。只有 CPU 标签不能满足 GPU 校验门槛。完整计划、逐题审核和配对比较可通过可选 JSON 导出查看。

<!-- END GENERATED OZ RESULTS -->

</details>

<details>
<summary>构造与独立 oracle</summary>

全部正式选例先比较独立参考，再做真实 T4 十次验证，检查形状、有限值、输入不变性、确定性与 hash。环境为 PyTorch 2.8.0、Triton 3.4.0、NumPy 1.26.4；CPU 仿真本身不能代替 GPU 真值。已评测公开代码和合同冻结，后来设计不改旧题。`oracle_source_sha256`记录原GPU冻结时的验证源码；本次仅整理路径，没有重新运行GPU或产生新实验。

| 机制 | 独立参考或额外检查 |
|---|---|
| attention、递推、求和 | FP64 与 scalar math.fsum；递推另用 Decimal80 与三种归约次序；求和用精确平衡的数学和 |
| 方差、线性求解、多项式 | 居中 FP64＋平移 fsum；NumPy solve＋Decimal80 消元；FP64 Horner＋Decimal80 直接幂次求和，均用实际存储 FP32 系数 |
| 正交投影、路由 | 重心化 FP64＋未重心化 Decimal80；未量化 FP64/Decimal80 距离、最小索引 tie-break，最终比较 embedding |
| 积分、频谱 | 连续解析积分＋Decimal80 三角级数；完整频谱 inverse FFT＋直接三角展开/fsum，Parseval 仅作旁证 |
| logdet、RBF | 实际 FP32 矩阵的 slogdet＋Decimal80 消元；直接差分平方后的完整 RBF 预测＋Decimal80，不能只比较中间距离 |

构造使用数值结果而非模型反馈。通过候选通常要求 ≤0.75 倍容差，失败 ≥1.25 倍；阈值校准与正式 seed 搜索分开记录。首次六题的日志包括 softmax 117 个候选、recurrence 512 个、reduction 3 个选中/草稿记录；后者不是完整候选规模的证明。扩展方差/求解各 256 个正式候选，分别 73/89 合格；多项式另有 128 个校准和 128 个正式候选，正式 91 合格。

| 后续搜索日志 | 校准合格/总数 | 正式合格/总数 | 正式 seed 范围 |
|---|---:|---:|---|
| [projection](private_data/search_log_orthogonal_projection.json) | 33/64 | 121/256 | 830100–830355 |
| [routing](private_data/search_log_quantized_routing.json) | 无单列校准 | 21/256 | 840100–840355 |
| [quadrature](private_data/search_log_quadrature.json) | 23/32 | 193/256 | 711000–711255 |
| [spectral](private_data/search_log_spectral_filter.json) | 17/32 | 126/256 | 811000–811255 |
| [logdet](private_data/search_log_logdet.json) | 38/100 | 135/256 | 98200–98455 |
| [distance](private_data/search_log_distance.json) | 139/200 | 192/256 | 119100–119355 |

Routing 另筛选真实前两名距离差 >1e-5、量化差 ≥1/64：210 候选符合，其中 19 合格，再按 seed 顺序选首个满足余量的正反例；其余五组取正式区间最小/最大误差。最终选中 seeds 为 830230/830228、840104/840101、711165/711076、811164/811158、98339/98262、119130/119263。所有生成器、独立参考和选择规则在 [families](../eval_scripts/single_call_vs_tools_challenges/families/)；完整日志比挑出的均衡题组更能说明选择偏差。

**早期探索记录不完整。** Logdet 最终 regularizer=1/1024、RBF 最终 offset=16 的校准行完整保留；此前 regularizer=1/64、1/256 和 offset=8、32 没有逐候选原始记录。2026-09-24T01:01:24Z 从更早工具输出补归档的摘要如下，不能当作恢复了原始候选日志或用来计算未保存配置的精确通过率：

| 早期配置 | seed 数/范围 | 最小 / 中位 / 最大误差 | 最小 / 最大 seed |
|---|---|---|---|
| logdet 1/64 | 100，98100–98199 | 1.8681e-8 / 4.7320e-6 / 3.3056e-5 | 98149 / 98188 |
| logdet 1/256 | 同上 | 3.2593e-7 / 2.9359e-5 / 1.1199e-4 | 98194 / 98144 |
| logdet 1/1024 | 同上 | 1.7046e-6 / 1.2300e-4 / 7.1892e-4 | 98157 / 98197 |
| RBF offset=8 | 200，118100–118299 | 5.6243e-6 / 0.00601084 / 0.0777140 | 118145 / 118197 |
| RBF offset=16 | 同上 | 0.000792979 / 0.0302787 / 0.335016 | 118135 / 118276 |
| RBF offset=32 | 同上 | 0.000127768 / 0.0916984 / 0.825148 | 118102 / 118167 |

</details>

<details>
<summary>证据审查与保留的问题</summary>

指定 extension 两轮的 24 条、oz 两轮的 48 条工具运行均有实际公开 kernel 的终值证据，误差与冻结 oracle 一致。审查核对源/合同、probe 文件、原始调用和 usage；不把正确 verdict 当作所有中间解释都正确。GPU 可用的工具调用也可能只执行 CPU 诊断；同一运行多个 probes 不算独立复验。

| 范围 | 保留的实质限制与原始证据 |
|---|---|
| 44–49 | [44 debate t12](../traces_glm/case_44/debate/r1/probes/t12_probe.py)把分母比值取反，6011% 诊断是假象；claims 已认定公式错误，直接输出比较及 t13 支持 trust。[47 debate t13](../traces_glm/case_47/debate/r1/probes/t13_probe.py)写 floor=0.04 而非0.004，本输入 norm=18.2166 使错误不生效；差两个误差范数也不能当舍入误差向量的范数。46 solo r1 输出 Python dict；[r2 t7](../traces_glm/case_46/solo/r2/probes/t7_probe.py)先打印有效测量再因 np.bool_ JSON 序列化 exit=1，未补跑，按保留 stdout 审核而非声称干净成功。44 debate/45 solo r2 的 CUDA→NumPy、48 debate r2 的 np.float32 JSON 错误在本运行内修复。 |
| 50–53 | 实测投影或最终 embedding；52 选 row6、53 真值 row9/候选 row7。[51 debate t13](../traces_glm/case_51/debate/r1/probes/t13_probe.py)把“精确 alpha”再转FP32，不能独立隔离逐元素舍入；r2仍需同样限定。53 的 tie_changes_selection 字段仅比较索引，不能证明存在 tie。53 solo 打印bug、52 debate CUDA→NumPy、51 debate r2占位模块import、53 debate r2参数/CUDA转换错误均保留并由后续真实GPU probe恢复；CPU tie分析不冒称GPU证据。 |
| 54–57 | 参考必须是连续积分/全部16模式。[54 debate r2 t13](../traces_glm/case_54/debate/r2/probes/t13_probe.py)把逐项误差矩阵Frobenius范数叫总积分误差，~0.0033解释不成立，t15终值有效。55 的角频率被写成Hz；其r2 signed差叫abs、shape检查硬编码、不同归一化范数比叫分解比例，均不用于主判定。57 r2 Parseval差约1.28e-8，不能写“within1e-8”。语法、np.sin参数、broadcast、Tensor.float64、缺import/变量和JSON序列化失败均留在trace，并有后续成功主probe；56的截断是否合格仍由全频谱误差决定。 |
| 58–59 | [58 debate r1 t15](../traces_glm/case_58/debate/r1/probes/t15_probe.py)测的是FP64主元转FP32，非真实FP32消元主元；t16编译失败未修复，c3保持inconclusive，t14终值独立支持trust。r2 t13成功获取真实主元，不回填r1失败；log_only等字段实际混入求和/消元差，59的log_accum_only更比较整条管线，不能仅按名字作因果分解。 |
| 60–61 | 60 r1 clamp检查是CPU仿真，61 r1 negative_raw_count误用了FP64对象。[60 debate r2 t12](../traces_glm/case_60/debate/r2/probes/t12_probe.py)在GPU调用后因CUDA→NumPy失败，t15才完成真实参考比较；61 r2修正clamp计数，但后续FP64 exp仿真与GPU仍差约7.55e-9，不是bitwise exact。61 solo r1被ledger阻止的重复probe没有发生第二次GPU执行。 |

多数运行有缺少 `scope_rationale` 等可恢复 ledger 错误，未删除失败文件；额外诊断有时未检查 dtype/shape/输入不变性，源码或私有冻结验证的检查不能冒称 agent 自己测量过。完整证据从 [GLM 索引](../traces_glm/INDEX.md)按 case/arm/rN 查看。

</details>

## 使用与结论边界

离线重建本页三个结果块：

```sh
python benchmark_fn_fp/eval_scripts/single_call_vs_tools_challenges/report.py
python benchmark_fn_fp/eval_scripts/single_call_vs_tools_challenges/report_extension.py
python benchmark_fn_fp/eval_scripts/single_call_vs_tools_challenges/report_oz.py
```

报告默认不生成额外 Markdown；`--json` 可导出 `private_data/reports/`。真实GPU oracle入口是 `modal run benchmark_fn_fp/eval_scripts/single_call_vs_tools_challenges/validate_modal.py`，会产生Modal费用。共享模型runner可用 `--dataset single_call_vs_tools_challenges --cases case_38,case_39`；执行模型实验另有API费用。已有trace不覆盖，已评测case不重建。

这些是主动挑选、数值均衡的合成有限工作负载，不是独立留出测试；相同输入重复衡量可重复性，不是新题泛化。固定参考脚本也能完成判定。下一步“给各臂相同初始可疑实验、比较参考独立性/覆盖/指标”的讨论发展为 [参考与覆盖组](../solo_vs_debate_challenges/README.md)；原提议中的 solo-review、debate-review、同成本独立多采样、结构化修复调度仍只是待验证方案，不能记为本组已经完成的额外实验。
