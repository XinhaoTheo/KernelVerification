# case_66/case_67: joint-distribution trace audit

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。范围为 `ea_methods_v2_r1` 的 case_66/case_67 × solo/debate 四次运行。
本审查只读取已经保存的完整 traces，并执行本地 CPU 哈希、代数和记录核验；没有新增 GPU 或模型请求。

**两种工具组均为 2/2 正确，没有 solo-wrong/debate-correct。Solo 各用一次真实 GPU 全域实验完成判断。Debate 的 case_66 最终标签正确，但多角色共同接受了错误的数学解释，因此不能把这条 trace 作为成功交叉纠错的证据。**

| 案例 / arm | 最终判断 | 决定性检查 | probe 次数 | API 请求 | 输入 / 输出 tokens | API 估算 |
|---|---|---|---:|---:|---:|---:|
| case_66 solo | trust，正确 | 实际 GPU 输出，全部 70×16 计数均为 64 | 1 | 5 | 60,352 / 1,485 | $0.018532 |
| case_66 debate | trust，正确 | 实际 GPU 输出，全部 70×16 计数均为 64 | 5（3 成功、2 Python 报错） | 10 | 412,286 / 14,379 | $0.131257 |
| case_67 solo | reject，正确 | 实际 GPU 输出，四元组 `[3,4,6,7]` 计数为 0/128 | 1 | 5 | 66,015 / 3,250 | $0.022059 |
| case_67 debate | reject，正确 | CPU 完整整数仿真＋GF(2) 依赖证明 | 2 | 9 | 260,804 / 8,389 | $0.082253 |

合计 29 个完整 API 请求，799,457 输入 tokens、27,503 输出 tokens，记录费用 $0.254101。
Solo 为 $0.040591，debate 为 $0.213510；费用依据项目现有 GLM Fireworks 定价估算，不含 Modal GPU。
共同配置为 GLM-5p3、`reasoning_effort=low`、每次运行累计输出上限 32,768；输入 tokens、费用和 GPU 时间没有匹配。

## 合同、冻结与主证据

合同要求固定的全部 1,024 个种子上，每个四通道子集的每种模式概率与 `1/16` 相差不超过 `1/1024`，等价于每个计数处于 `[63,65]`。共 70 个子集、每组 16 个模式。初始测试只有一阶和二阶直方图，因此两例的真实初始输出都是通过。

[冻结 GPU 记录](private_data/validation_gpu.json)保存了每例十次完整域执行：case_66 最大概率偏差为 0；case_67 为 0.0625，仅 `[3,4,6,7]` 不满足合同。二者重复输出一致，输入未改动，dtype、shape、finite、输出只能为 0 或 `2X` 均符合。公开源码/合同、四份 `trace_meta.json` 的哈希及四份 `run.json` 中实际提供的源码/合同全部一致。

- **case_66 solo**：[t7 代码](../traces_glm/case_66/solo/r1/probes/t7_probe.py)、[t7 输出](../traces_glm/case_66/solo/r1/probes/t7_stdout.txt)、[verdict](../traces_glm/case_66/solo/r1/verdict.json)。使用真实 `make_inputs()` 和 `run()`，枚举全部四元组；测得误差 0，另行验证结构约束和输入不变性。其 `passes_contract` 布尔表达式没有合并所有结构检查，但各检查分别输出为 true，因此本次结论有完整支持。
- **case_67 solo**：[t7 代码](../traces_glm/case_67/solo/r1/probes/t7_probe.py)、[t7 输出](../traces_glm/case_67/solo/r1/probes/t7_stdout.txt)、[verdict](../traces_glm/case_67/solo/r1/verdict.json)。直接从真实 GPU 输出恢复 keep bits，完整计算四阶直方图，另算二阶、三阶分布。失败四元组的完整计数与冻结结果逐项一致；对应 masks `30 ^ 667 ^ 324 ^ 961 == 0`。
- **case_66 debate**：[t16 代码](../traces_glm/case_66/debate/r1/probes/t16_probe.py)、[t16 输出](../traces_glm/case_66/debate/r1/probes/t16_stdout.txt)、[t17 输出](../traces_glm/case_66/debate/r1/probes/t17_stdout.txt)、[verdict](../traces_glm/case_66/debate/r1/verdict.json)。成功主实验确实调用真实 kernel，全部 1,120 个计数均为 64。t13 也调用 kernel，但只打印被错误五维矩阵标记的 33 个子集；它单独不能覆盖全部合同。输入不变性没有由这些 probes 测量；源码只向新分配的 Out 写入，可独立支持输入不变。最终文字把它描述为实验已通过，超出了实际测量范围。
- **case_67 debate**：[t12 代码](../traces_glm/case_67/debate/r1/probes/t12_probe.py)、[t12 输出](../traces_glm/case_67/debate/r1/probes/t12_stdout.txt)、[t13 代码](../traces_glm/case_67/debate/r1/probes/t13_probe.py)、[t13 输出](../traces_glm/case_67/debate/r1/probes/t13_stdout.txt)、[verdict](../traces_glm/case_67/debate/r1/verdict.json)。这两个 probe **没有调用候选 GPU kernel**。它们用相同 PCG64 种子重建 masks/offsets，正确顺序执行 XOR-fold，并完整枚举固定种子域。仿真用 int64，候选用 int32；本合同数值均在非负十位范围，位操作等价，不存在溢出或符号扩展差异。四阶计数、依赖子集和 affine offset XOR=1 与独立私有 oracle、实际冻结 GPU 结果一致。这是有效的结构分析和完整有限域 CPU 证据，不能标为这次 agent 自己跑出的 GPU 四阶实验。

## case_66 debate：正确标签与错误解释同时存在

[Skeptic 的 c1/c2/c3](../traces_glm/case_66/debate/r1/claims.json)把顺序更新
`f ^= f >> 8; f ^= f >> 4; f ^= f >> 2; f ^= f >> 1`
错误当成所有右移都作用于原始 `f`，声称只保留输入位 `{0,1,2,4,8}`，因此只需分析一个 8×5 矩阵。
这是错误的：顺序 XOR-fold 会传播中间结果，正确计算十位输入的 parity。例如仅原始 bit 3 为 1 时，最终 parity 仍为 1。

[t13](../traces_glm/case_66/debate/r1/probes/t13_probe.py)确实测得这个错误删列矩阵有 33 个秩不足的四元组，但它并非实际 kernel 的线性映射。实际完整 8×10 mask 矩阵秩为 8，全部四元组秩为 4；t16 自己已经输出 **256 个不同 keep vectors**，也直接否定“五维输入至多 32 个输出”的前提。

Experimenter 面对所有 GPU 计数均为 64，未修正错误的线性模型，而声称 affine offsets 使非齐次依赖重新覆盖全部模式。[Judge 最终理由](../traces_glm/case_66/debate/r1/verdict.json)和末轮 [Skeptic review](../traces_glm/case_66/debate/r1/run.json)接受了这一说法。它在数学上不成立：对输出 XOR 一个固定 offset 是双射，只平移线性像，不能增加秩或可达模式数。通过的真实原因是原先删掉的位本来就在顺序 parity 中起作用。

因此，该 trace 展示的是实验阻止了错误 reject，但跨角色复核没有纠正错误因果解释；solo 的一次有效实验已经得到相同正确标签。这里没有额外的 debate 准确率优势。

## 其他局部问题与失败保留

**case_67 solo 的辅助 `formula_match=false` 源自 probe 自身写错。** t7 中的 `keep_ref` 使用一次性 `f ^ (f>>8) ^ (f>>4) ^ (f>>2) ^ (f>>1)`，没有顺序更新。它没有复现候选算法，不能作为 kernel 位运算错误的证据。决定性的四阶计数取自真实 GPU 输出，未依赖此错误辅助参考，因此 reject 有效。

**case_67 debate 的一般性代数命题过强。** c1 rationale 和 c2 statement 声称八通道中任何非空依赖都会破坏四阶独立。实际只需排除大小不超过四的依赖；五阶及以上依赖不必破坏四阶合同。一个直接反例是 masks `[1,2,4,8,16,32,64,127]`：全体秩为七，唯一依赖包含八个通道，所有四元组仍独立。本次找到的确实是四通道依赖，所以该过强概括未改变 case_67 结论。t13 的注释将“某个子集 XOR 非零”与“该子集所有行独立”等同，也不是一般成立的等价关系。

case_66 debate 的 [t12](../traces_glm/case_66/debate/r1/probes/t12_stderr.txt)在不存在坏样例时执行 `list(None)` 报错，[t14](../traces_glm/case_66/debate/r1/probes/t14_stderr.txt)索引 `itertools.combinations` 迭代器报错；两次都在真实 kernel 执行之后失败，后续 t16/t17 修复输出代码。失败 trace 保留，不能把它们计为成功证据。两个 debate 还各遇到两次 `scope_rationale` 遗漏的可恢复 ledger 错误，随后补齐。

## 原始记录与预算完整性

四次运行的 **29 个 request、29 个 response、29 份 usage** 全部存在，调用数与 history 中有 usage 的模型轮数一致。逐调用检查 model、reasoning_effort，以及 `max_tokens == 32768 - 此前累计 completion_tokens`；全部符合，每个 run 的累计输出不超过 32,768，没有 length 截断或 provider 错误。保留了 9 个 probe 尝试，其中 7 个成功、2 个失败；case_67 debate 的 2 个成功 probe 是 CPU 分析，其余 7 个尝试包含真实 GPU 执行。

工具事件中引用的 **34 个 probe 代码/stdout/stderr/JSON 文件哈希**全部核验一致，包括失败文件；其中 28 个文件属于最终 claims 引用的成功证据。没有追加模型重试、修改公开案例或改写已保存的模型解释。
