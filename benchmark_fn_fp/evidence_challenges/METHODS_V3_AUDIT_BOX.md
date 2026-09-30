# case_72/case_73: continuous-box trace audit

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。审查 `ea_methods_v3_r1/r2` 的 case_72/case_73 × single_call/solo/debate 十二次运行；没有新增模型或 GPU 请求，没有修改公开案例和冻结真值。下面先保留 r1 审查，再追加 r2。

**两轮合计：无工具 2/4 正确、1 次明确错误、1 次弃答；solo 4/4，debate 4/4。本 pair 没有 solo-wrong/debate-correct。两种工具系统都发现了凸残差的完整顶点证书，并取得真实 GPU 测量。Solo 没有停在随机采样通过这一层。**

## r1 结果

| 案例 / arm | 判断 | 主要依据 | API 请求 | 输入 / 输出 tokens | API 估算 |
|---|---|---|---:|---:|---:|
| case_72 single_call | reject，错误 | 推测共中心符号会同时激活多个神经元，没有计算固定参数最大值 | 1 | 1,434 / 6,061 | $0.007069 |
| case_72 solo | trust，正确 | 采样后主动升级为真实 GPU 的全部 4,096 个顶点＋凸性论证 | 9 | 145,877 / 3,738 | $0.044957 |
| case_72 debate | trust，正确 | CPU 全部顶点＋真实 GPU 行对齐顶点检查 | 7 | 172,407 / 6,480 | $0.055402 |
| case_73 single_call | reject，正确 | 定性共中心激活推断，没有具体数值反例 | 1 | 1,435 / 1,567 | $0.002125 |
| case_73 solo | reject，正确 | 全顶点搜索后执行真实 GPU 反例，误差 1.34375 | 5 | 67,173 / 2,766 | $0.021851 |
| case_73 debate | reject，正确 | 全顶点搜索后执行同一个真实 GPU 反例 | 6 | 138,755 / 7,685 | $0.047305 |

r1 合计 29 个请求、527,081 输入 tokens、28,297 输出 tokens，API 估算 $0.178709。其中无工具 $0.009194、solo $0.066808、debate $0.102707；不含 Modal GPU。三组使用 GLM-5p3、low reasoning 和每次运行累计输出上限 32,768；输入 tokens、费用和 GPU 时间未匹配。

## 合同和独立冻结证书

令 `R(x)=sum_j c[j]*ReLU(W[j]·x-b[j])`。固定参数满足 `c[j]=0.25>0`，因此 R 非负且凸，在 `[-1,1]^12` 上的最大值等于 4,096 个顶点的最大值。单纯的有限采样通过不能推出全域通过；完整顶点检查结合这一结构证明可以确定 R 的全域最大值。

此外，实际 kernel 的 FP32 线性 base 与数学 base 可能有舍入差异，真实绝对误差不能未经说明直接等同于 R(x)。[冻结证书](private_data/validation_gpu.json)独立使用两种方法：FP64 全顶点计算，以及 64 个 active-subset 的整数支持函数最大化；结果完全一致。再计入两次二进制幂缩放和一次加法的舍入/FTZ 保守界 `1e-7`：

| 案例 | R 的精确最大值 | 实际 kernel 的全域误差上界 | 逐神经元独立最大值之和 |
|---|---:|---:|---:|
| case_72 | 0.71875 | 0.7187501 | 3.40625 |
| case_73 | 1.34375 | 1.3437501 | 3.3125 |

case_72 的上界小于阈值 1；case_73 在合法顶点取得真实误差 1.34375，足以拒绝。两个 kernel 都在 T4 上重复完整顶点测量十次，结果一致。初始 41 行 smoke 的最大误差均为 `2.9802322387695312e-08`。四个工具 run 提供的源码、合同和元数据哈希，以及两份无工具用户 prompt，均与公开文件和冻结记录一致。

## 决定性证据

**case_72 solo**：[t7 代码](../traces_glm/case_72/solo/r1/probes/t7_probe.py)、[t7 输出](../traces_glm/case_72/solo/r1/probes/t7_stdout.txt)先尝试行对齐、随机内部点和随机符号角点，找到误差 0.71875；它没有把该采样结果当作最后依据。随后 [t11 代码](../traces_glm/case_72/solo/r1/probes/t11_probe.py)、[t11 输出](../traces_glm/case_72/solo/r1/probes/t11_stdout.txt)把全部 4,096 个顶点作为合法 batch 输入真实 kernel，使用实际 FP32 参数转 FP64 算完整目标，最大误差仍是 0.71875。[最终 verdict](../traces_glm/case_72/solo/r1/verdict.json)明确给出凸性推广，因此它并非只做有限点经验检查。其论证遗漏单独写出 base 舍入界；冻结证书中的 `1e-7` 可补齐这一步，且距阈值的余量为 0.28125。

**case_72 debate**：[t9 代码](../traces_glm/case_72/debate/r1/probes/t9_probe.py)、[t9 输出](../traces_glm/case_72/debate/r1/probes/t9_stdout.txt)在 CPU FP64 中完整枚举顶点，结合凸性得到 R 最大值 0.71875。此 probe **没有运行 kernel**。[t11 代码](../traces_glm/case_72/debate/r1/probes/t11_probe.py)、[t11 输出](../traces_glm/case_72/debate/r1/probes/t11_stdout.txt)实际运行六个行对齐顶点，最大误差同为 0.71875，结构约束与输入不变性检查正确。[Judge](../traces_glm/case_72/debate/r1/verdict.json)提到了约 `3e-8` 的 FP32 舍入，但把 R 的最大值直接称为全域实际误差精确最大值仍不够严谨；安全充分结论是上界 `0.7187501<1`。末轮 Skeptic 说线性 base “trivially exact” 也不适用于所有合法 FP32 内部点，真实 smoke 自己已显示非零舍入。

**case_73 solo**：[t7 代码](../traces_glm/case_73/solo/r1/probes/t7_probe.py)、[t7 输出](../traces_glm/case_73/solo/r1/probes/t7_stdout.txt)用 CPU 完整顶点搜索确定合法反例 `[-1,1,-1,1,1,-1,1,-1,1,1,1,1]`，随后把该点复制成四行，真实运行 kernel。输出相对 FP64 目标的误差是 1.34375，且输入未改动，shape/dtype/finite 全部满足。反例与冻结最大值一致，已独立足够支持 [reject](../traces_glm/case_73/solo/r1/verdict.json)。

**case_73 debate**：[t8](../traces_glm/case_73/debate/r1/probes/t8_probe.py)和 [t9](../traces_glm/case_73/debate/r1/probes/t9_probe.py)是 CPU 分析，正确重建实际 seed 194003 的 FP32 参数并发现同一最大值；t9 还查到仅一个顶点超过阈值 1。[t12 代码](../traces_glm/case_73/debate/r1/probes/t12_probe.py)、[t12 输出](../traces_glm/case_73/debate/r1/probes/t12_stdout.txt)真实执行该点，得到输出 0.25、目标 1.59375、误差 1.34375。[最终 reject](../traces_glm/case_73/debate/r1/verdict.json)有直接有效反例支持；不需要证明它也是全域最大值才能拒绝。

## 论证与实验的局部限制

case_72 solo t11 的 `before = clone(inputs)` 写在 kernel 已经执行之后，再与当前输入比较，因此其 `inputs_unmodified=True` 是无效的实验检查。不能把这行输出当成输入未修改的测量。源码实际只向新分配 output 写入，且冻结验证与初始 probe 都正确在运行前保存输入，所以最终标签不受影响。

case_72 solo 在完全枚举之前，把 t7 的采样证据登记为 `rebutted`，语义过强；当时的采样不能否定“存在任一坏输入”。不过它紧接着完成了完整顶点检查，因此最终 evidence 已补足。t8 尝试由于上一 probe 尚未解释被 ledger 拒绝，没有执行 GPU；一次 finalize 调用又因 `status=inconclusive` 与 `supports=rebutted` 冲突被拒绝。随后修正并继续执行 t11。不能把 t8 计作实际实验。

case_73 solo [t6 报错](../traces_glm/case_73/solo/r1/probes/t6_stderr.txt)源于用三行反例 output 对比原始 41 行 smoke target；case_72 debate [t8 报错](../traces_glm/case_72/debate/r1/probes/t8_stderr.txt)源于 CUDA 输入与 CPU 权重混合矩阵运算。两次都实际执行过 kernel，随后在参考计算/比较中失败；后续成功 probe 修复，失败文件保留。

case_73 debate 的 c2 rationale 假设“全域最大值低于 1，但某个近顶点超过 1”，这在定义上不可能；所谓翻转一两个符号的“近顶点”也仍是已完整枚举的顶点。它作为额外 borderline 假设最终被驳回，没有影响有效反例，但没有提供超出完整顶点搜索的覆盖范围。

无工具两例都用共中心结构推测多个神经元可以同时激活。case_72 的这种断言被完整最大值 0.71875 反驳；case_73 虽然猜对 reject，却没有给出固定参数的可核验数值反例。两者 [case_72 verdict](../traces_glm/case_72/single_call/r1/verdict.json)、[case_73 verdict](../traces_glm/case_73/single_call/r1/verdict.json)按原协议分别计明确错误和正确，不因缺少证明把正确标签改成错误。

## r1 原始 traces 与预算

全部 **29 份 request、29 份 response、29 份 usage** 完整；四个工具 run 的 raw 调用数与 history usage 数一致。逐调用核对 GLM-5p3、low reasoning、`max_tokens == 32768 - 此前累计实际输出 tokens`，六个 run 均合规，未出现 length 截断、累计预算耗尽或 provider 错误。两个无工具 run 各恰好一个请求，未提供 tools，公开源码/合同均完整包含在 prompt。

共有 11 次 `run_claim_probe` 请求：1 次 ledger 阻止执行；其余 10 次中 8 次成功、2 次 Python 报错。其中 7 次包含真实 GPU kernel 调用（包括报错前已执行的两次），3 次为 CPU 分析。工具事件引用的 **38 个 probe 文件哈希**全部吻合，失败文件也计入核验。

这对案例检验了从有限 smoke 测试升级为全域结构证明的能力。Solo 已经主动完成这一步，两种工具系统的主证据都有效，这两例自身没有提供 solo-wrong/debate-correct。v3 其他机制的触发结果需另看总报告；以上结论仅针对本 pair 的 r1。

## r2 复测结果

r2 由 v3 其他机制的预定触发条件启动，按协议重复全部四例、全部三组；这对 box 案例也保留全部槽位，没有根据第一轮答案挑选重测。

| 案例 / arm | 判断 | 主要依据 | API 请求 | 输入 / 输出 tokens | API 估算 |
|---|---|---|---:|---:|---:|
| case_72 single_call | trust，正确 | 认为同时高度对齐多个随机行不太可能，没有计算具体最大值 | 1 | 1,434 / 9,392 | $0.010733 |
| case_72 solo | trust，正确 | CPU 全顶点证书＋真实 GPU 最大点 | 5 | 60,482 / 2,173 | $0.019325 |
| case_72 debate | trust，正确 | CPU 全顶点证书＋真实 GPU 最大点 | 9 | 163,097 / 5,351 | $0.051553 |
| case_73 single_call | needs_more_evidence | 明确需要计算固定 PCG64 参数的联合激活，未给二元判断 | 1 | 1,435 / 5,465 | $0.006413 |
| case_73 solo | reject，正确 | 真实 GPU 的 2,006 个合法候选点中找到误差 1.34375 | 5 | 61,079 / 2,313 | $0.019646 |
| case_73 debate | reject，正确 | 真实 GPU 全部 4,096 个顶点，误差 1.34375 | 10 | 246,244 / 7,244 | $0.076917 |

r2 共 31 个请求、533,771 输入 tokens、31,938 输出 tokens，API 估算 $0.184587。无工具 $0.017146、solo $0.038971、debate $0.128470。所有二元工具判断再次正确；无工具 case_73 是弃答，没有算作明确错误。

**case_72 solo r2**：[t7 代码](../traces_glm/case_72/solo/r2/probes/t7_probe.py)、[t7 输出](../traces_glm/case_72/solo/r2/probes/t7_stdout.txt)、[verdict](../traces_glm/case_72/solo/r2/verdict.json)。正确重建实际种子参数，CPU 穷举完整顶点，并在真实 GPU 上验证最大点；结果 0.71875。额外 200,000 个随机内部点的残差全部为 0，说明它们不能替代顶点证书。这 200,000 行只用于 CPU 参考搜索，传给候选 kernel 的 batch 是合法 n=1。最终“base 完全精确，因为只有两个 elementwise operations”的说法错误；与 r1 一样需要补充 `1e-7` 全域舍入界。probe 未测量新反例输入的不变性或直接检查 dtype/shape；它引用初始 smoke 的结构检查，源码静态读写也支持结构要求。

**case_72 debate r2**：[t10 代码](../traces_glm/case_72/debate/r2/probes/t10_probe.py)、[t10 输出](../traces_glm/case_72/debate/r2/probes/t10_stdout.txt)、[verdict](../traces_glm/case_72/debate/r2/verdict.json)。`meshgrid` 生成了完整 4,096 个符号组合，另加原点只在 CPU 参考中计算；GPU batch 是最大点复制八行，满足 n 上限。真实误差为 0.71875，输入在调用前 clone，immutability 检查有效。完整顶点＋凸性得到 R 的全域值，仍需舍入余量把它转为实际 kernel 的充分上界。随机符号点和六个行对齐点对已有完整顶点枚举没有增加覆盖范围。

**case_73 solo r2**：[t7 代码](../traces_glm/case_73/solo/r2/probes/t7_probe.py)、[t7 输出](../traces_glm/case_73/solo/r2/probes/t7_stdout.txt)、[verdict](../traces_glm/case_73/solo/r2/verdict.json)。六个行对齐点加 2,000 个随机符号点组成合法 n=2,006 batch；直接调用 kernel 并从实际 output 与 FP64 目标之差找到 1.34375 的合法反例，已充分支持 reject，无需完整全域枚举。但它把逐神经元最大值相加得到的 3.3125 称作“worst-case dropped residual”，最终理由也称“analytic worst-case”；这是松上界，不是实际共同 x 可达到的最大值，真实值为 1.34375。此术语错误没有污染用于拒绝的实际测量。stdout 使用多行格式化 JSON，框架没有保存派生 JSON-result 文件；完整 stdout 和人工填写的 evidence 数值一致，没有丢失原始测量。

**case_73 debate r2**：[t12 代码](../traces_glm/case_73/debate/r2/probes/t12_probe.py)、[t12 输出](../traces_glm/case_73/debate/r2/probes/t12_stdout.txt)、[verdict](../traces_glm/case_73/debate/r2/verdict.json)。真实 kernel 接收全顶点矩阵，最大误差仍为 1.34375。这里按低位先行生成顶点，最坏索引 3930 对应的坐标与冻结 witness 相同；不能因为索引和其他枚举顺序不同而判断输入变了。三个激活 margin 为 2.125、2.125、1.125，乘 0.25 后之和为 1.34375。

t12 的 `inputs_unmodified` 是**硬编码 true**，并没有前后比较；Judge 与末轮 Skeptic 称已验证输入不变，超出了该 probe 的证据。真实源码和独立冻结验证仍证明输入没有被修改，因此 reject 本身有效。[t15](../traces_glm/case_73/debate/r2/probes/t15_probe.py)只在 CUDA PyTorch 中计算参考残差，未调用候选 kernel，确认 smoke 残差为 0、仅一个顶点超过 1。其 `smoke_miss_factor=1.34375e12` 来自把零分母替换为 `1e-12` 的人为 floor，不能当成自然定义的误差倍率。此前 [t13 报错](../traces_glm/case_73/debate/r2/probes/t13_stderr.txt)将 `.max()` 绑定到 c 张量，造成矩阵乘标量错误；后续 t15 修复，失败保留。

## 两轮完整性与结论范围

r2 的 31 个 raw request/response/usage 全部完整，model、low reasoning、逐请求递减的累计输出预算、源码/合同冻结哈希均符合协议；两个 source-only prompt 包含完整共同材料且没有 tools。四份 run history 的 usage 数与 raw 请求数一致，22 个工具 artifact 哈希全部匹配。六次 probe 执行中五次成功，一次辅助参考报错；四个工具 run 都实际运行了候选 kernel，另两个 case_73 debate probes 是参考计算。没有 length/provider/budget failure。五次缺少 `scope_rationale` 的 ledger 记录请求都在原 trace 中保留并随后修复。

两轮合计 **60 个完整 API 请求/响应/usage、60 个经核验的 probe 文件**，1,060,852 输入 tokens、60,235 输出 tokens，API 估算 **$0.363296**（无工具 $0.026340、solo $0.105779、debate $0.231177），不含 GPU。

这对连续域案例没有重复出现 debate 的额外标签优势。两种工具系统都能用结构证明关闭有限测试的覆盖缺口；同时，粗上界被误叫真实最大值、舍入被忽略、输入不变性被伪检查等论证质量问题仍需单独审查。不能用正确最终标签替这些辅助论证背书，也不能把本 pair 的平局覆盖到 v3 其他机制。
