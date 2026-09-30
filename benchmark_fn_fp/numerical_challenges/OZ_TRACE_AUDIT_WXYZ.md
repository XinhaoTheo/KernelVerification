# case_58–case_61 工具证据审查

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

r1 完成审查时间：2026-09-24T01:06:34Z；r2 全部完成审查时间：2026-09-24T01:15:41Z。最终范围为 W、X、Y、Z 在 `oz_low32_r1`、`oz_low32_r2` 的 solo 与 debate 共 **16/16 条工具运行**。只读检查 canonical traces，并在本地复算哈希与 CPU 参考；未修改案例、运行 GPU 或发起模型请求。早期 01:01:24Z 快照中八个 r1 槽位尚只有预留元数据；随后两轮完整运行均已收集。

**16/16 条最终标签都有正确的原始内核 GPU 终值 probe 支撑，且终值与冻结 T4 oracle 逐位一致。** 共保存并核验 112 个原始模型调用，r1 为 58、r2 为 54。部分补充诊断的描述过强，且 case_58 debate r1、case_60 debate r2 各有一个失败 probe；这些限制与恢复过程如下单独保留，不能把“最终判对”写成“所有中间解释都已证明”。

## 原始内核的终值证据

下表误差为合同的原始数值，未乘以 100。每个主 probe 都调用对应公开模块原有 `run()`，执行的是 CUDA Triton 内核。`output/ref` 在同组两条 trace 中一致。

| 案例 | 工具组 | 主 probe 代码与结果 | output / reference | 实测合同误差 | 容差 | 最终 verdict | 原始 API 调用数 |
|---|---|---|---|---:|---:|---|---:|
| case_58 | solo | [t7 代码](../traces_glm/case_58/solo/r1/probes/t7_probe.py)、[结果](../traces_glm/case_58/solo/r1/probes/t7_json_result.json) | 24.4865074158 / 24.4865024608 | 2.0235508513e-7 | 0.0001 | trust | 5 |
| case_58 | debate | [t14 代码](../traces_glm/case_58/debate/r1/probes/t14_probe.py)、[结果](../traces_glm/case_58/debate/r1/probes/t14_json_result.json) | 同上 | 2.0235508513e-7 | 0.0001 | trust | 9 |
| case_59 | solo | [t7 代码](../traces_glm/case_59/solo/r1/probes/t7_probe.py)、[结果](../traces_glm/case_59/solo/r1/probes/t7_json_result.json) | 25.7928752899 / 25.8112508355 | 0.0007119199948 | 0.0001 | reject | 5 |
| case_59 | debate | [t13 代码](../traces_glm/case_59/debate/r1/probes/t13_probe.py)、[结果](../traces_glm/case_59/debate/r1/probes/t13_json_result.json) | 同上 | 0.0007119199948 | 0.0001 | reject | 9 |
| case_60 | solo | [t7 代码](../traces_glm/case_60/solo/r1/probes/t7_probe.py)、[结果](../traces_glm/case_60/solo/r1/probes/t7_json_result.json) | 0.299753963947 / 0.299711338376 | 0.0001422220832 | 0.05 | trust | 5 |
| case_60 | debate | [t12 代码](../traces_glm/case_60/debate/r1/probes/t12_probe.py)、[结果](../traces_glm/case_60/debate/r1/probes/t12_json_result.json) | 同上 | 0.0001422220832 | 0.05 | trust | 9 |
| case_61 | solo | [t7 代码](../traces_glm/case_61/solo/r1/probes/t7_probe.py)、[结果](../traces_glm/case_61/solo/r1/probes/t7_json_result.json) | -0.0185366719961 / -0.0331569900446 | 0.292406360971 | 0.05 | reject | 6 |
| case_61 | debate | [t12 代码](../traces_glm/case_61/debate/r1/probes/t12_probe.py)、[结果](../traces_glm/case_61/debate/r1/probes/t12_json_result.json) | 同上 | 0.292406360971 | 0.05 | reject | 10 |

case_58 的参考是实际输入转 FP64 后的 `np.linalg.slogdet`；case_59 使用 `np.log(np.linalg.det(A64))`。后者通常不如 slogdet 稳健，但本例 determinant 有限且为正，参考与独立 Decimal80 结果相差仅 2.50e-12，足以区分这里的阈值。case_58 的对应参考差为 5.18e-12。两个 logdet probe 都正确使用 `abs(out-ref)/max(abs(ref),1)`。

case_60/case_61 的参考由实际存储的 FP32 输入升为 FP64，从 `(anchors-query)**2` 直接差分距离出发，计算指数权重及最终归一化预测。它们没有把易相消的 norm expansion 当作参考，也没有只用中间距离误差下结论。参考与 Decimal80 的差分别为 5.55e-17、4.16e-17；均正确使用分母 `max(abs(reference),0.05)`。

## 来源、文件与原始调用核验

- 对八份 `run.json` 内保存的 `kernel_code`、`problem_text` 重新计算 SHA256，均等于相应 `trace_meta.json`、冻结 [private_data/validation_gpu.json](private_data/validation_gpu.json) 及当前公开文件的 hash。
- 从保存的公开源码独立提取并执行 NumPy 输入生成器，得到的每个输入 hash 均匹配冻结 oracle。主 probes 使用这个生成器或逐字一致的种子/生成步骤，没有另造有利输入。probe 未单独打印运行时输入 hash；这里的证据链是冻结源码、确定性公开生成器及重建输入 hash，不能说每条 probe 都自行核验了输入 hash。
- 把八条主 probe 输出恢复为 FP32 单元素数组，字节 SHA256 均匹配相应冻结 T4 输出，合同误差也与 oracle 一致。部分 probe 打印了 shape、dtype 和 finite，部分只返回标量 `.item()`；后者没有独立打印所有属性，但冻结 `run()` 的分配代码定义了 `(1,)` 和 FP32，且返回的实际有限终值与 oracle 相符。
- 所有保存的 probe 代码、stdout、stderr、已解析 JSON 文件均与 `run.json` 中记录的 artifact hash 一致，包括 case_58 debate 的失败诊断 stderr。
- 八条共 58 个模型调用目录（solo 21、debate 37），每个都有完成状态的原始 request/response/metadata，响应均带 usage。请求统一记录 `accounts/fireworks/models/glm-5p3`、`reasoning_effort=low`、`max_tokens=32768`。没有 runner 级错误；这并不意味着中间所有工具操作都成功。

## 失败、恢复及补充诊断限制

**四条 solo 都发生一次 claim 参数错误。** 各自 t5 的 `record_claim` 缺少 `scope_rationale`，记录为可恢复 `LedgerError`；t6 补齐字段后成功。case_61 solo 还在 t8 尝试重复执行尚未 finalize 的成功 probe，被 ledger 拒绝，随后 t9 正确解释 t7、t10 给出 verdict。t8 没有产生第二次 GPU 执行，不能作为独立复验。相关记录完整保留在各自 `tool_events.jsonl` 和 `run.json`。

**case_58 debate 的主结论有效，但两项解释需要收窄。** t6–t8 的三次 claim 缺字段错误，在 t9–t11 恢复。主 t14 成功运行原始内核，直接满足固定输入合同。补充 [t15](../traces_glm/case_58/debate/r1/probes/t15_probe.py) 测试 `tl.log` 时，使用的是 **FP64 消元所得主元再转换为 FP32**，不是原内核实际 FP32 消元所得主元。独立 CPU 复算可见八个主元中有四个不同，例如末项分别为 0.004353681113570929 与 0.004353702068328857。因此它仅证明 `tl.log` 在所测替代主元上匹配 `np.log(float64).astype(float32)`，不能支持最终 reason 中“全部真实主元已逐位验证”的说法。

case_58 debate 随后的 [t16](../traces_glm/case_58/debate/r1/probes/t16_probe.py) 试图从另写的 Triton 变体提取真实主元，但 `matrix[N-1,N-1]` 触发不支持 constexpr 整数 tensor 索引的[编译错误](../traces_glm/case_58/debate/r1/probes/t16_stderr.txt)。exit code=1、无 JSON 结果，未修复重跑；c3 被保留为 `inconclusive`。最终 trust 仍有 t14 的原始内核终值依据，但“任何错误 mask 都不可能产生如此小误差”不是这个实验能证明的普遍命题，也不需要该命题才能判断此固定合同。

**case_59 debate 区分了实际内核运行与 CPU 模拟，但诊断字段命名过窄。** t12 是 FP32 消元的 CPU 模拟；t13 才是原始内核 GPU 执行。t13 把 GPU FP32 累加输出与模拟主元的 FP64 log 求和相比，差 1.7301e-6；这个差同时包含对数近似和求和舍入，不能严格命名为单独的 `tl.log` 误差。它可以支持“后续 log/累加差远小于总失配”的诊断，实际 reject 由 t13 的合同误差直接支持。其 t5 描述更新参数混入损坏的键名，t6–t7 claim 缺字段；后续 t8–t10 修复，原始错误均保留。

**case_60 debate 的补充 clamp 检查是 CPU 模拟。** t12 明确返回 `gpu_used=true`、`device=cuda:0` 并调用原始内核。t13 顺序模拟 FP32 accumulation，发现所有 16 个 pre-clamp 距离都为正，最小 0.001953125；这一辅助结果不应称为 GPU 中间值采样。无论是否依赖这项解释，t12 的最终误差已经充分支持 trust。t6–t7 claim 参数错误后在 t8–t9 修复。

**case_61 debate 的一个诊断计数字段计算对象不正确。** t12 正确执行 GPU 主内核并计算最终误差，也另做 FP32 距离模拟，但 `negative_raw_count` 实际使用 FP64 norm expansion 来计数，不能据它判断 FP32 clamp 次数。不过返回的 16 个 FP32 模拟、clamp 后距离均严格为正，这另行支持本输入没有被截为零。应修正的是诊断解释范围；最终 reject 的 GPU 合同证据不受影响。t5 描述更新缺 `summary`、t6–t7 claim 缺字段，t8–t10 后恢复。

本节审查确认的是八条 r1 工具运行的终值证据与数据完整性；不把多个同运行辅助 probe 算作预注册的独立重复，不把每个最终 reason 都视为完全准确。r2 及无工具/default 对照仍应按 [OZ_PROTOCOL.md](OZ_PROTOCOL.md) 与持续更新的 [OZ_REPORT.md](OZ_REPORT.md) 单独统计。

## r2 补审：八条全部完成

首次补审时间为 2026-09-24T01:12:04Z，覆盖四条 solo 与 case_58 debate；最后三条 case_59/case_60/case_61 debate 于 01:15:41Z 完成补审。本节只检查新收集的 `oz_low32_r2` traces，未重新运行构造、模型或 GPU。

| 案例 | 工具组 | 主 GPU probe | 合同误差 | 最终 verdict | 已保存 API 调用数 |
|---|---|---|---:|---|---:|
| case_58 | solo | [t7 代码](../traces_glm/case_58/solo/r2/probes/t7_probe.py)、[结果](../traces_glm/case_58/solo/r2/probes/t7_json_result.json) | 2.0235508513e-7 | trust | 5 |
| case_59 | solo | [t7 代码](../traces_glm/case_59/solo/r2/probes/t7_probe.py)、[结果](../traces_glm/case_59/solo/r2/probes/t7_json_result.json) | 0.0007119199948 | reject | 5 |
| case_60 | solo | [t7 代码](../traces_glm/case_60/solo/r2/probes/t7_probe.py)、[结果](../traces_glm/case_60/solo/r2/probes/t7_json_result.json) | 0.0001422220832 | trust | 5 |
| case_61 | solo | [t7 代码](../traces_glm/case_61/solo/r2/probes/t7_probe.py)、[结果](../traces_glm/case_61/solo/r2/probes/t7_json_result.json) | 0.292406360971 | reject | 5 |
| case_58 | debate | [t12 代码](../traces_glm/case_58/debate/r2/probes/t12_probe.py)、[结果](../traces_glm/case_58/debate/r2/probes/t12_json_result.json) | 2.0235508513e-7 | trust | 9 |
| case_59 | debate | [t12 代码](../traces_glm/case_59/debate/r2/probes/t12_probe.py)、[结果](../traces_glm/case_59/debate/r2/probes/t12_json_result.json) | 0.0007119199948 | reject | 9 |
| case_60 | debate | [t15 代码](../traces_glm/case_60/debate/r2/probes/t15_probe.py)、[结果](../traces_glm/case_60/debate/r2/probes/t15_json_result.json) | 0.0001422220832 | trust | 10 |
| case_61 | debate | [t8 代码](../traces_glm/case_61/debate/r2/probes/t8_probe.py)、[结果](../traces_glm/case_61/debate/r2/probes/t8_json_result.json) | 0.292406360971 | reject | 6 |

八条都调用冻结公开模块的原始 `run()`；实际输出恢复为 FP32 后的字节 hash 与冻结 oracle 一致，误差与 oracle 一致。case_58 solo 用 FP64 `log(det(A))`，case_59 solo 与 case_58/case_59 debate 用 FP64 `slogdet(A)`；case_60/case_61 两组工具都使用 FP64 直接差分距离、指数权重及最终归一化预测。所有分母与阈值符合合同，实际结果足以支持相应最终标签。部分 probe 未单独打印 dtype/shape，相关证据界限与 r1 相同，不能把仅返回有限标量描述为主动验证了所有形状断言。

再次核验八条保存源码、合同、当前文件与 GPU 冻结记录的 hash 一致；所有新 probe artifact hash 均与工具记录一致，包括 case_60 的失败 stderr。用于表中终值判断的主 probes 都 exit=0、未超时。r2 的 54 个原始 API 调用全部有已完成 request/response/metadata 及 usage，模型和 low/32768 配置一致；无 runner 级错误。四条 solo 都在 t5 因缺 `scope_rationale` 被拒、t6 修复；case_58/case_59/case_60 debate 的同类错误为 t6/t7，t8/t9 修复，case_61 debate 未出现工具参数错误。这些恢复没有被隐去或算作新的实验。

**case_58 debate r2 的主元诊断比 r1 更完整。** t12 对原始内核进行终值测试，并做 CPU FP32 消元分解；随后 [t13](../traces_glm/case_58/debate/r2/probes/t13_probe.py) 成功执行保存主元和 log 值的 Triton 变体，使用与原始内核一致的更新规则、`div_rn`、单 warp 和禁用融合设置，返回的末主元为 0.004353702068328857，与 CPU FP32 消元一致。这个新实验没有重现 r1 的 constexpr 索引编译失败，也没有用 FP64 主元代替 FP32 主元。它另调用原始 `run()`，终值误差仍为 2.0235508513e-7。

t13 的 `max_abs_per_pivot_log_diff=0` 支持“这些采样主元上的 `tl.log` 与 FP64 log 后转 FP32 的结果相同”。但 `log_only_rel_err=9.7367e-9` 实际比较 FP32 求和与 FP64 求和，既然逐项 log 值相同，这个非零差来自求和精度，仍不应称为独立 log 近似误差。该命名问题不影响主终值合同判断。r2 的成功诊断单独记录，不能回填为 r1 当时失败 probe 的成功证据；同一 r2 运行里的 t12/t13 也只计一个预注册实验槽位。

**case_59 debate r2 的总误差证据成立，分解字段仍不能按名字解释。** t12/t13 均执行原始 GPU 内核；[t13](../traces_glm/case_59/debate/r2/probes/t13_probe.py) 的 FP32 重放得到同一个终值，FP64 消元重放与参考相对差约 1.67e-13。不过 `log_accum_only_error=0.0183755` 比较的是整个 FP32 管线与 FP64 管线，包含主元消元、log 及累加差，不能当作只隔离了 log/累加的误差，更不能仅靠此比较证明全部失配都来自某一个步骤。主 GPU 合同误差为 0.00071192，远超 0.0001，reject 有直接依据。

**case_60 debate r2 保存了失败 probe，并在同一运行中修复。** [t12](../traces_glm/case_60/debate/r2/probes/t12_probe.py) 已调用原始内核，但之后直接对 CUDA tensor 使用 `.numpy()`，在计算参考前触发 [TypeError](../traces_glm/case_60/debate/r2/probes/t12_stderr.txt)；exit=1，无 JSON 比较结果。不能把它视为一次成功数值核验。t13 的 CPU FP32 距离模拟先用于排除 clamp，随后 t15 给所有 tensor 增加 `.cpu()` 再计算参考，成功返回表中误差，并在 t16 finalize。失败与修复代码、stderr、结果均保留；没有用未记录的重试替换原槽位。其最终关于权重接近因而误差相消的描述是解释性归因，真正充分的依据是 t15 的最终预测比较。

**case_61 debate r2 正确核验 clamp，但最终 reason 的“精确重现”应改为近似一致。** t8/t9 都运行原始内核。[t9](../traces_glm/case_61/debate/r2/probes/t9_probe.py) 使用逐步 FP32 算术得到 pre-clamp 距离，16 个值都为正，最小 0.001953125，修正了 r1 计数用错 FP64 对象的问题。但它在距离之后使用 FP64 exp、权重和求和，`emulated_clamp_output=-0.018536664444918166` 与实际 GPU `-0.018536671996116638` 相差约 7.55e-9，既非完整 FP32 管线模拟，也非 bitwise exact。差异很小且不改变 reject 标签；仍应保留这个证据范围，而非照抄最终 reason 中“exactly”。

最终审查范围完整覆盖预注册的 case_58–case_61 两轮工具组 16 个槽位。每例的 solo 两次、debate 两次都有可审核的正确终值证据；辅助 CPU 模拟、重复 GPU probe、失败后修复都保留在所属运行内，未增加独立实验计数。本审查不涵盖无工具/default 控制的内容审查，也不据此推断 debate 超过 solo。
