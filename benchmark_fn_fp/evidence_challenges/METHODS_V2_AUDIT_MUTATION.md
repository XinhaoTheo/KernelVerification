# case_70/case_71 methods-v2 trace audit

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。范围：`ea_methods_v2_r1` 的 case_70/case_71 × solo/debate，共四次带工具运行。只审查已经保存的源文件、冻结真值、原始 API 请求/响应和执行证据；没有追加 GPU 或模型调用。

**四次最终判断均正确，主实验均遵守“全部更新完成后观察保留对象”的合同。Solo 已经独立识别并正确测试返回值别名问题，本组没有观察到 debate 的额外判断优势。**

| 案例 / arm | 最终判断 | 主实验历史误差 | 执行 probes | API 请求 | 输入 / 输出 tokens | API 估算 |
|---|---|---:|---:|---:|---:|---:|
| case_70 solo | trust，正确 | 0.005744695547110313 | 1 | 5 | 65,614 / 2,415 | $0.021028 |
| case_70 debate | trust，正确 | 0.005744695547110312 | 3 | 9 | 208,207 / 6,343 | $0.065275 |
| case_71 solo | reject，正确 | 0.8989803703905964 | 2，含一次失败 | 6 | 83,722 / 2,575 | $0.026275 |
| case_71 debate | reject，正确 | 0.8989803703905964 | 2 | 9 | 199,134 / 5,764 | $0.062098 |

历史误差阈值为 `0.025`，最终参数和动量的相对误差阈值均为 `1e-5`。费用合计 **$0.174676**，其中 solo $0.047303，debate $0.127373；来自原始 usage 与项目 GLM 价格配置，不含 GPU。

## 决定性证据和观察时间

- **case_70 solo**：[t7 probe](../traces_glm/case_70/solo/r1/probes/t7_probe.py)、[实际输出](../traces_glm/case_70/solo/r1/probes/t7_stdout.txt)、[最终判断](../traces_glm/case_70/solo/r1/verdict.json)。从未更新的实际 FP32 输入保存 FP64 数值，独立递推时为每个目标历史值做 `.copy()`；原样调用 `kernel.run_sequence`，等函数完成才 `torch.stack(hist)`。候选返回值没有在更新之间被克隆。历史误差、最终状态、梯度原始字节、形状和有限性均通过。最终结论明确指出别名本身不构成失败，实际误差小于容差。
- **case_70 debate**：[t14 历史 probe](../traces_glm/case_70/debate/r1/probes/t14_probe.py)、[结果](../traces_glm/case_70/debate/r1/probes/t14_stdout.txt)采用相同的正确观察顺序：先保存独立输入，再运行完整候选序列，最后才转换和堆叠候选历史。参考每步独立 `.clone()`，不存在共同别名。[t15 最终状态](../traces_glm/case_70/debate/r1/probes/t15_stdout.txt)和 [t16 梯度字节](../traces_glm/case_70/debate/r1/probes/t16_stdout.txt)各用新鲜输入验证另外两项合同。三个 probe 都有实际 GPU 执行，分别覆盖历史、最终状态和禁止修改的梯度。
- **case_71 solo**：[t8 probe](../traces_glm/case_71/solo/r1/probes/t8_probe.py)、[结果](../traces_glm/case_71/solo/r1/probes/t8_stdout.txt)、[最终判断](../traces_glm/case_71/solo/r1/verdict.json)。更新前保留参数和动量，调用原始 `run_sequence` 后才堆叠原返回对象；`.float()` 位于整个序列结束之后，不会修复候选历史。独立 FP64 参考对每个目标值克隆，测得 89.8980% 历史误差，超过 2.5% 阈值。最终状态相对误差约 `5.05e-8` / `4.51e-8`，梯度与重新生成的规范输入相等，输出形状和有限性正常。
- **case_71 debate**：[t12 历史 probe](../traces_glm/case_71/debate/r1/probes/t12_probe.py)、[结果](../traces_glm/case_71/debate/r1/probes/t12_stdout.txt)先建立独立 FP64 逐步目标，再运行候选，明确同步后堆叠保留对象。历史误差与冻结真值一致。[t13 状态与梯度 probe](../traces_glm/case_71/debate/r1/probes/t13_probe.py)、[结果](../traces_glm/case_71/debate/r1/probes/t13_stdout.txt)在另一份新鲜状态上验证最终参数/动量、float32 dtype、有限性、形状和梯度字节不变。[最终判断](../traces_glm/case_71/debate/r1/verdict.json)正确指出初始 probe 的参考也别名化，因此初始零误差不能推翻独立失败证据。

四份主历史测量与[冻结 GPU 真值](private_data/validation_gpu.json)一致，仅末位浮点舍入有约 `1e-16` 差异。所有实验使用合同的固定种子、六步更新和正确的 `max(norm(target), 0.1)` 分母；没有增加范围外输入，也没有修改候选实现。所有候选对象均在完整六步事务后才被观察，独立参考却保留各步数值：这里区分了两种必须不同的存储语义。

## 失败和证据限制

case_71 solo 的首次 [t7 probe](../traces_glm/case_71/solo/r1/probes/t7_probe.py)已经执行 GPU 序列，但在诊断 `torch.equal(H[t], w.cpu())` 时混用了 CUDA 与 CPU 张量，[运行失败](../traces_glm/case_71/solo/r1/probes/t7_stderr.txt)。此失败未被当作合同反例；后续 t8 从新鲜输入修复设备比较，得到有效结果。原失败代码和 stderr 均保留。

四次运行还各遇到遗漏 `scope_rationale` 的可恢复 claim 参数错误；后续均修复。Debate 的 Skeptic 在最终 verdict 前复核了完成的证据，没有要求把容差合同改成“禁止别名”。

个别检查有表达上的小限制：case_70 solo 把候选转换成 double 后报告形状/有限性，没有单独输出原 dtype；其返回路径和实际 float32 参数提供了静态佐证。case_71 solo 使用 `torch.equal(g, regenerated_g)` 而非 uint8 视图比较，其最终“byte unchanged”措辞比该表达式直接展示的检查更强。这个固定输入中的梯度为有限、非零 FP32 数值，且公开 kernel 对梯度只有读取；私有冻结验证也进行了实际字节比较。没有发现这些限制影响任一最终标签或关键历史误差。

## Raw traces、冻结和预算核验

- 四次运行共 **29 个 API 请求、29 个响应、29 份完整 usage**，与 `run.json` 内带 usage 的 history 数量逐一一致；没有 `finish_reason=length`、遗漏响应或传输失败。
- 所有实际请求均为 `accounts/fireworks/models/glm-5p3`、`reasoning_effort=low`。逐调用验证 `max_tokens <= 32768 - 此前累计实际输出`；四次累计输出分别为 2,415 / 6,343 / 2,575 / 5,764，均在 32,768 预算内。相同输出上限不等于相同输入 tokens、费用或 GPU 时间。
- `trace_meta.json`、`run.json` 内实际提供的源码和合同、当前公开文件的 SHA256，均与冻结 GPU 记录一致。
- 核验全部 **31 个执行证据文件哈希**，涵盖八次 probe 的代码/stdout/stderr 和七份成功解析的 JSON；其中 claim 引用的 28 个文件也全部匹配。失败 t7 的三个文件未被遗漏。

作为旁证，无工具单次调用对两例都给出 `reject`：case_70 明确误拒，case_71 标签正确。case_70 的无工具解释从“所有历史均为末步值”直接跳到“必然超容差”，没有证明数值量级；两个工具系统都通过正确观察时间的实际测量消除了该错误。本轮结果支持工具的价值，未支持 debate 超过 solo。该结论仅覆盖此成对构造的一轮预先固定实验。
