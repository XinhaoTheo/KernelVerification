# Fresh-seed precision transfer: case_79–case_81 trace audit

2026-09-24。范围：`ea_precision_transfer_r1` 的 case_79/case_80/case_81 × 三个 arm，共九次运行。审查仅读取保存的源文件、请求/响应、实验、结果和冻结真值，没有追加付费调用或修改案例。

**这三个案例中，solo 与 debate 最终标签均为 3/3 正确，无工具为 1/3。case_79 solo 虽然标签正确，但其决定性误差计算使用错误参考；这一证据质量差异应独立记录，不能事后改变预先规定的标签评分。**

| Case | 真值 / 真实相对误差 | 无工具 | Solo | Debate |
|---|---|---|---|---|
| case_79 | reject / 0.661099152757181 | trust，错误 | reject，标签正确；参考错误 | reject，正确且证据有效 |
| case_80 | trust / 5.725813846601292e-8 | trust，标签正确 | trust，正确且证据有效 | trust，正确且证据有效 |
| case_81 | reject / 1.0 | trust，错误 | reject，正确且证据有效 | reject，正确且证据有效 |

所有真值均来自[冻结 T4 验证](private_data/validation_gpu.json)：每例十次实际 GPU 执行可重复，输出与逐步 FP32 Neumaier 模拟一致，数学目标由 `math.fsum` 和独立精确 dyadic 整数求和交叉核验。固定合同阈值为 `1e-5`，不改变输入范围或评价指标。

## case_79：相同正确标签，不同证据质量

**Solo 的 [t7](../traces_glm/case_79/solo/ea_precision_transfer_r1/probes/t7_probe.py)错误地把普通 NumPy FP64 reduction 视为精确求和。** 它使用 `xf.sum(axis=1)`，声称输入能被 FP64 精确表示且最终和很小，所以该求和精确；这个推导忽略中间的 `2^80` 累加器。实际[输出](../traces_glm/case_79/solo/ea_precision_transfer_r1/probes/t7_stdout.txt)中的所谓 target 为全零，伪相对误差为 `5.806527899328642e12`。[最终 reject](../traces_glm/case_79/solo/ea_precision_transfer_r1/verdict.json)恰好与真值一致，但该比较本身无法证明合同失败。

**Debate 的 [t11](../traces_glm/case_79/debate/ea_precision_transfer_r1/probes/t11_probe.py)使用实际 GPU 输出和精确 Fraction 目标。** [结果](../traces_glm/case_79/debate/ea_precision_transfer_r1/probes/t11_stdout.txt)为真实目标 `[7.970721662044525, 7.673868119716644, 8.070558607578278, 8.786630541086197]`，候选输出 `[2.184875965118408, 3.43967866897583, 3.508208751678467, 2.191612482070923]`，相对误差 0.661099152757181；dtype、shape、finite 和输入不变性均检查通过。该结果逐项匹配冻结记录，独立支持 reject。

Debate 的 [t12](../traces_glm/case_79/debate/ea_precision_transfer_r1/probes/t12_probe.py)又运行 kernel 并以 CPU FP32 模拟复核，[两者输出完全一致](../traces_glm/case_79/debate/ea_precision_transfer_r1/probes/t12_stdout.txt)。这支持当前 workload 的输出是算法自身行为。其 PTX 辅助检查则只报告三个 opcode 计数均为零，没有保存 PTX 文本或检查字符串非空，而且 `add.f32` 并不覆盖所有带修饰符的指令拼写。因此不能用该诊断证明完整编译代码中没有相关操作或所有潜在编译器问题；决定性的正确参考比较不依赖此诊断。

无工具错误 trust：它正确怀疑初始 FP64 参考，却把 Neumaier 的 correction 当成总能以 FP32 精度保留全部小量，忽略 correction 本身未经进一步补偿，在 `±2^30` 期间仍可丢失小量。

## case_80：solo 也能可靠识别有问题的参考

Solo 的 [t7](../traces_glm/case_80/solo/ea_precision_transfer_r1/probes/t7_probe.py)实际运行 kernel，将存储值转为对象数组中的 Python float 后用 Fraction 精确求和。其[完整 stdout](../traces_glm/case_80/solo/ea_precision_transfer_r1/probes/t7_stdout.txt)记录正确目标、GPU 输出、误差 `5.725813846601292e-8` 以及结构和不变性检查。[trust](../traces_glm/case_80/solo/ea_precision_transfer_r1/verdict.json)有充分证据，说明不能把 case_74 的两次 solo 错误推广为所有新 seed 都会失败。它打印的是 Python dict 而非 JSON；数值证据保留在 stdout，不应错误声称有已解析 JSON 文件。

Debate 的 [t8](../traces_glm/case_80/debate/ea_precision_transfer_r1/probes/t8_probe.py)也实际执行 GPU，使用精确 Fraction 求和，得到[相同误差和输出](../traces_glm/case_80/debate/ea_precision_transfer_r1/probes/t8_stdout.txt)。[t9](../traces_glm/case_80/debate/ea_precision_transfer_r1/probes/t9_probe.py)仅在 CPU 重算精确目标和初始顺序 FP64 参考，确认后者全零及约 `1.6e13` 的初始伪误差。两个实验独立支持最终 trust；t9 不是第二次 GPU 执行。

辅助字段 `per_row_abs_err` 实际保存带符号差值，注释“exact rational norm”实际使用 FP64 norm；名称和注释过强，但不影响符合合同的 FP64 相对 L2 计算。

无工具同样返回 trust，但理由是概括性的“有补偿就能精确恢复”，未判断固定排列下 correction 是否丢失小量。因此此处标签正确，不能把该理由视为已经验证了这一固定输入的误差界。

## case_81：修复错误参考后正确拒绝，仍有 ledger 缺口

Solo 第一次 probe 在真实 GPU 执行后因 `Fraction(np.float32)` 类型错误失败，随后用 [t8](../traces_glm/case_81/solo/ea_precision_transfer_r1/probes/t8_probe.py) 修复。其[结果](../traces_glm/case_81/solo/ea_precision_transfer_r1/probes/t8_stdout.txt)显示 GPU 全零输出，对精确目标 `[7.428201526403427, 9.405435025691986, 6.688341170549393, 6.113687425851822]` 的真实误差为 1.0，支持 reject。该 probe 验证 finite 和输入不变性，未显式打印原始 dtype/shape；后两项还可由公开 launcher 和冻结实验支持，不应把它们说成此 probe 独立测量过。

Debate 的 t12 初次 GPU probe 和 t13 CPU 机制模拟都把普通 FP64 reduction 的零当成 exact target。模型随后发现问题，并在 [t17](../traces_glm/case_81/debate/ea_precision_transfer_r1/probes/t17_probe.py) 真正修复：重跑实际 kernel，以仅包含八个小量的 FP64 求和为参考，核对 RNG 生成的小量与实际输入提取值一致，再用完整行的 `math.fsum` 独立交叉验证。[t17 输出](../traces_glm/case_81/debate/ea_precision_transfer_r1/probes/t17_stdout.txt)与冻结目标逐项一致，误差为 1.0。因此最终 reject 有有效数值证据。

成功 t17 没有 finalize 到 c1，后者仍 inconclusive；[最终判断](../traces_glm/case_81/debate/ea_precision_transfer_r1/verdict.json)改以已确认的 c2 机制和原始 t17 结果支持 reject。这里确有完整的正确实验，区别于没有执行修复的情况，但证据登记缺口应保留。t13 的普通 FP64 `exact_target` 字段依然是错误参考，不能因为它附在 confirmed claim 下就视为真值。正确机制是八个小量全在 `±2^30` 之前：两大项相邻并不足以保证通过，小量也可能在第一大项到来时从 correction 中被抹去。

无工具再次错误 trust，认为所有丢失项都能由 correction 精确保留，与实际已提供的零输出及固定输入的正目标不符。

## 原始记录与预算检查

- 九次运行共 **45 个原始 API 请求、45 个响应及 45 份 usage**，与模型 history 逐次记录一致；合计输入 **893,643**、输出 **41,554** tokens。
- 每个请求都是 `accounts/fireworks/models/glm-5p3`、`reasoning_effort=low`；逐次核对 `max_tokens = 32768 - 本次运行此前累计输出`，没有预算耗尽、长度截断、API 传输失败或弃答。
- 公共源码/合同、实际 prompt 或 run artifact、metadata 哈希与冻结 GPU 记录全部一致。三次无工具运行各有一个 API 请求，没有工具调用。
- 已核验记录引用的 **42 个去重实验代码、stdout、stderr、JSON 文件哈希**，包含保留的失败实验。case_80 solo 的 Python dict stdout 没有解析 JSON，coverage 检查未要求伪造不存在的文件。
- 本文九次运行 API 估算 **$0.29592944**，不含 GPU，也不是六个 transfer 案例的总费用。相同累计输出上限不意味着实际调用数、费用或输入 token 相同。

此子集没有 solo/debate 正确率差距，但展示了 case_79 的证据质量差异和 case_80 的可靠 solo 反例。是否满足完整 cohort 的下一轮条件，应以全部六个案例及[预先规定的协议](PRECISION_TRANSFER_PROTOCOL.md)计算，不能只选择这一子集或改用事后证据质量评分。
