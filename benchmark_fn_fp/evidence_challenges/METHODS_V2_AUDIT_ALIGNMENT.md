# case_68/case_69 shared-alignment pilot trace audit

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。范围：`ea_methods_v2_r1` 的 case_68/case_69 × solo/debate，共四次带工具运行。审查读取保存的源文件、原始 API 请求与响应、probe、结果和冻结 T4 真值；没有追加模型调用或 GPU 实验。

**四次最终判断均正确，且均直接执行实际 GPU kernel，穷举合同允许的 384 个共享变换。Solo 已自行区分局部对齐与全局对齐；这对案例没有产生 debate 的额外判断优势。**

| Case / arm | 最终判断 | 全局合同误差 E | 执行 probes | API 调用 | 输入 / 输出 tokens | API 估算 |
|---|---|---:|---:|---:|---:|---:|
| case_68 solo | trust，正确 | 0 | 1 | 5 | 62,533 / 1,633 | $0.019306 |
| case_68 debate | trust，正确 | 0 | 2 | 9 | 216,354 / 6,032 | $0.067214 |
| case_69 solo | reject，正确 | 0.8316711011500714 | 2 | 5 | 66,345 / 1,575 | $0.020309 |
| case_69 debate | reject，正确 | 0.8316711011500714 | 2 | 6 | 151,096 / 4,887 | $0.047683 |

四次 API 估算合计 $0.154512，不含 GPU。Probe 数包含 case_69 solo 一个运行后打印失败的实验，以及 case_68 debate 一个仅检查 CPU anchors 的辅助实验。

## 合同与决定性实验

合同要求一个共享列排列 `p` 和共享符号向量 `s`，同时解释两个 view 的全部八行。分母为 `max(norm(features), 1e-12)`，分子为整个 `(2,8,4)` 张量的 Frobenius 误差，阈值 0.05。所有主实验都使用规定 `make_inputs()`、实际 `run()` 输出和 FP64 计算，没有把每个 view 分别最小化后再合并，也没有放宽为任意旋转。

- **case_68 solo**：[t7 代码](../traces_glm/case_68/solo/r1/probes/t7_probe.py)、[输出](../traces_glm/case_68/solo/r1/probes/t7_stdout.txt)、[最终判断](../traces_glm/case_68/solo/r1/verdict.json)。穷举 `24 × 16` 个共享变换，得到 `p=[0,2,1,3]`、`s=[1,1,-1,-1]`、E=0；同时检查输出 dtype、shape、finite 和两个输入未修改。
- **case_68 debate**：[t12 主实验](../traces_glm/case_68/debate/r1/probes/t12_probe.py)、[输出](../traces_glm/case_68/debate/r1/probes/t12_stdout.txt)、[最终判断](../traces_glm/case_68/debate/r1/verdict.json)。相同完整穷举得到同一个共享解与 E=0；明确承认零列带来的局部非唯一性。[t13](../traces_glm/case_68/debate/r1/probes/t13_probe.py)只在 CPU 检查固定 anchors 的绝对值无 ties，未额外执行 kernel；[结果](../traces_glm/case_68/debate/r1/probes/t13_stdout.txt)与固定输入一致。
- **case_69 solo**：[t7 代码](../traces_glm/case_69/solo/r1/probes/t7_probe.py)、[输出](../traces_glm/case_69/solo/r1/probes/t7_stdout.txt)、[最终判断](../traces_glm/case_69/solo/r1/verdict.json)。完整穷举得到最小 E=0.8316711011500714，远高于阈值；最佳共享解为 `p=[1,2,0,3]`、`s=[1,-1,-1,-1]`。输出和输入不变性检查均通过。
- **case_69 debate**：[t8 主实验](../traces_glm/case_69/debate/r1/probes/t8_probe.py)、[输出](../traces_glm/case_69/debate/r1/probes/t8_stdout.txt)、[最终判断](../traces_glm/case_69/debate/r1/verdict.json)。也穷举整个有限变换集，得到相同最小值与最佳共享解。[t9 辅助实验](../traces_glm/case_69/debate/r1/probes/t9_probe.py)另一次调用实际 kernel，检查 anchors、slots 和带符号的源列匹配；[结果](../traces_glm/case_69/debate/r1/probes/t9_stdout.txt)支持无 tie、slot 排列完整。

四个主实验报告的误差和最佳变换都与[冻结 T4 验证](private_data/validation_gpu.json)一致。冻结验证还用独立的带符号赋值代价 oracle 复核，并重复实际 kernel 10 次。模型 probes 没有打印完整输出张量或其哈希，因此这里核对的是合同误差与最佳变换，不能宣称逐字节核对了模型 probes 的输出张量。

## 局部非唯一性处理

case_68 的初始 probe 给两个 view 选择了不同符号代表，但差异所在源列恰好在对应 view 为零。每个 view 有八个精确局部最优解，两个 view 仍有共同可行解。Solo 和 debate 都重新求解共享问题，正确保留 trust，没有用“局部代表不同”作为拒绝理由。

case_69 的两个 view 也各自精确匹配，但它们没有低于阈值的共同变换。两个系统都直接计算全局最小值，正确拒绝。结论来自明确的误差值，并非仅凭候选按 view 计算 anchors 这一源码模式。

## 保留的失败与辅助解释限制

- case_69 solo 的 [t6](../traces_glm/case_69/solo/r1/probes/t6_probe.py)实际运行了 GPU kernel 和穷举，然后因打印字典键没有引号触发 [NameError](../traces_glm/case_69/solo/r1/probes/t6_stderr.txt)。模型随后用 t7 修复，得到完整 JSON。失败文件完整保留；t6 不是一个有输出数值的成功证据。
- case_68 solo 和 debate 曾分别遗漏 `record_claim` 的必填 `scope_rationale`，后续在原运行内修复。没有 API 传输失败、输出上限截断或 runner 失败。
- case_69 debate t8 的 `inputs_unmodified` 实际只检查了 features，未比较 anchors 的前后值；它把输出转为 FP64 后仅报告 shape/finite，没有直接记录原始 output dtype。其最终“inputs unmodified”以及“没有其他合同问题”的表述比该 probe 覆盖范围更宽。源码和冻结验证支持这两个属性，但模型自身实验记录不能视为完整独立验证。E 已超阈值，reject 不依赖这些额外判断。
- case_69 debate t9 注释声称填充 NaN sentinel，实际未填充，而是直接调用 `K.run` 后用 `np.allclose` 检查各源列拷贝。不能把它当作真正的 sentinel 写覆盖实验。固定 anchors 的 slots 恰为完整排列，加上源码中的对应 store，仍支持该固定 workload 不存在由 anchor tie 导致的漏写；主全局误差判断不受影响。
- case_68 debate 的辅助 claim 将 anchor tie 后的任意未初始化内容直接表述为“E cannot be <= 0.05”，这个泛化的必然性过强。实际实验只证明当前 workload 没有 tie；并未验证任意有 tie 输入都会超阈值。最终判定始终限定固定输入。

## Trace 与预算完整性

- 共 **25 个原始 API 请求、25 个响应、25 份 usage**，与模型 history 的逐次输入/输出 token 记录完全一致；合计输入 496,328、输出 14,127 tokens。
- 每个请求均为 `accounts/fireworks/models/glm-5p3`、`reasoning_effort=low`。逐次核验 `max_tokens = 32768 - 此前累计输出`，四次运行都未耗尽累计输出预算；所有响应 finish reason 为 `tool_calls` 或 `stop`。
- 四次 `trace_meta.json`、`run.json` 中实际提供的 kernel/contract 哈希均与冻结 GPU 记录一致；当前公开输入材料包含真实初始 probe 输出，不含私有标签。
- 核验 claims 引用的 **24 个去重 probe/code/stdout/stderr/JSON 文件哈希**，全部匹配。另有 case_69 solo t6 的失败代码和 stderr 保留于同一 trace。
- 预算匹配仅指累计输出 token 上限；输入 token、实付费用和 GPU 调用数不同。每例每 arm 只有这次预先规定运行，不能据此估计稳定的错误率或宣称泛化优势。

该机制确实需要从局部证据回到全局合同，但本轮 solo 一次完整穷举已经解决。Debate 增加了辅助检查和讨论，没有改变最终正确率，并仍保留上述未被末轮复核纠正的辅助证据表述问题。
