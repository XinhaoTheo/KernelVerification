# case_72–case_75 independent oracle review

存储命名更新（2026-09-30）：本文批次标签及 r1/r2 轮次简称保留原实验含义，原标签记在 `trace_meta.json.original_trial`。统一目录使用每个 case/arm 下的 `rN`；见[命名与迁移说明](../TRACES.md)。

2026-09-24。本审查独立于构造者，只读取现有 family、公开源码、合同和完整 CPU 搜索记录，并另外实现精确有理数计算作交叉核验。没有修改案例、追加 GPU 调用或调用模型。本文件记录 **CPU 真值与数学证明审查**，以及主流程随后保存的 GPU 冻结记录的只读核验。

## 审查结果

已选 case_72–case_75 的数学真值和适用输入域均有明确依据，没有发现阻断 GPU 验证的问题。唯一发现的非选中边界记录已修正为未定：box seed `194063` 的最大残差恰好等于容差，现记录 `requires_boundary_analysis`，没有将无法证成的上界误当作失败证据。

| 案例 | 固定 seed | 数学/CPU 判定 | 决定性数值 | 初始 probe 的局限 |
|---|---:|---|---|---|
| case_72 | 194001 | trust | 最大残差 0.71875；含 base 舍入的全域上界 0.7187501 < 1.0 | 41 个 smoke 点不能单独证明全盒域；本例实际有完整凸性证书 |
| case_73 | 194003 | reject | 顶点反例误差 1.34375 > 1.0 | smoke 最大误差仅约 2.98e-8，漏掉联合激活 |
| case_74 | 193662 | trust | FP32 仿真相对误差 1.9235792803862544e-8 < 1e-5 | 顺序 FP64 参考丢失小项，错误地拒绝 |
| case_75 | 193602 | reject | FP32 仿真相对误差 1.0 > 1e-5 | 候选与顺序 FP64 参考都给出零，错误地通过 |

随后已读取[实际 GPU 冻结记录](private_data/validation_gpu.json)：四例均有十次可重复的 T4 验证、输入不可变检查，实际误差分别为 0.71875、1.34375、1.9235792803862544e-8、1.0，与表中 CPU 结果一致。case_74/case_75 的实际输出与 FP32 仿真逐位一致；case_72/case_73 的全部 4,096 个顶点也匹配候选 base 仿真。此处核验已有记录，没有追加实际执行。

## case_72/case_73：整个输入盒上的误差证书

审查材料：[family](families/box_relu_audit.py)、[完整 64-seed 搜索](private_data/search_box_relu.json)、[私有选中记录](private_data/answer_key_box_relu.json)、[case_72 合同](eval_cases/case_72/problem.txt)、[case_73 合同](eval_cases/case_73/problem.txt)。

写残差为 `R(x) = Σ_j c_j max(w_j·x - b_j, 0)`。本构造所有 `c_j = 1/4 > 0`，因此 R 非负且凸。在 `[-1,1]^12` 上，凸函数的最大值等于某个顶点值；原搜索枚举全部 **4,096 个顶点**，覆盖每个候选的残差最大值。此结论覆盖整个实数盒，因而也覆盖合同允许的所有存储为 FP32 的点，而非仅覆盖有限抽样。

独立恒等式为：

`R(x) = max_{S subset {1,...,6}} Σ_{j in S} c_j(w_j·x-b_j)`。

对每个子集 S，线性函数在盒内的最大值为：

`||Σ_{j in S} c_j w_j||_1 - Σ_{j in S} c_j b_j`。

审查另写了 Python `Fraction` 版本，逐项读取存储的权重、偏置、系数，不调用 family 的 `independent_certificate()`，以有理数计算所有子集支持值。对 **64 个候选 × 64 个激活子集，共 4,096 个精确子集计算**取最大值，结果全部匹配原搜索保存的顶点最大值。原构造另采用分母 32 的 int64 表达；数值范围极小，没有整数溢出。空子集包括在内，恒等式没有遗漏所有单元关闭的情况。

这里不能把“残差是凸函数”直接套到任意 `abs(candidate-target)` 上。证书成立还依赖两个已检查条件：

1. 目标为线性 base 加非负 R，而候选仅计算相同的 base。
2. 实际 FP32 base 与数学 base 的绝对误差在整个盒内有独立上界。

候选 base 是 `0.25*x[0] + 0.5*x[1]`。正常范围内二进制幂缩放精确，和的绝对值不超过 0.75，最后一次 FP32 舍入最多约 `2^-25 = 2.98e-8`；极小值的下溢或 subnormal flush 误差远小于 `1e-7`。因此 family 使用的 `BASE_ROUNDOFF_BOUND = 1e-7` 是足够保守的绝对界，且

`abs(candidate-target) <= R(x) + 1e-7`。

case_72 的 0.71875 最大残差给出明确通过的全域界。case_73 的最大值在 ±1 顶点达到；顶点处 base 的乘加本身精确，因此 1.34375 是实际算术候选的明确反例误差，不依赖上界的松紧。每个合法批次仅对行独立应用相同运算，源码掩码正确覆盖 `1 <= n <= 4096`，只向输出写入。

case_72/case_73 的逐单元最大值均不超过 0.59375，但逐单元最大值之和分别为 3.40625 / 3.3125。逐单元上界之和不是联合可达值，不能据此拒绝 case_72；完整联合优化才区分这一对案例。

### 边界记录的修正

初版 `_candidate` 把 `M + 1e-7 > 1` 的候选直接标为 reject。对 seed `194063`，`M == 1`；该条件仅说明当前上界无法证明通过，并未证明存在超阈值输入。审查指出此问题，并另外检查两个最大值顶点周围，分别把 x[0]/x[1] 向盒内移动 0–16 个相邻 FP32 值，共 `2 × 17 × 17 = 578` 个点，最大仿真误差仍恰好 1.0。这个额外局部搜索也不能证明整个盒通过，故正确状态是未定。

已重新读取当前 family 与 `private_data/search_box_relu.json`，确认 seed `194063` 已改为 **`requires_boundary_analysis`**。已选 case_72/case_73 使用远离阈值的通过/失败余量，未受该修正影响；本审查没有把 578 点局部搜索当成完整边界证明。

## case_74/case_75：精确数学和，而非普通 FP64 累加

审查材料：[family](families/precision_oracle_audit.py)、[完整 256-seed 搜索](private_data/search_precision_oracle.json)、[私有选中记录](private_data/answer_key_precision_oracle.json)、[case_74 合同](eval_cases/case_74/problem.txt)、[case_75 合同](eval_cases/case_75/problem.txt)。

每行包含相反的 `±2^80` 端点、相反的 `±2^30` 内部项，以及八个存储为 FP32 的正小项。数学目标是这些存储值的精确实数和，和具体累加顺序无关。普通顺序 FP64 累加在大数附近丢失小项，不能充当这里的真值。FP32 Neumaier 补偿的 correction 自身也以 FP32 累加，是否保留所有小项取决于顺序；因此补偿算法名称本身也不能决定标签。

私有第一参考为 `math.fsum`。第二参考将每个 FP32 精确转换为整数/二次幂分母，提升到共同分母后使用 Python 任意精度整数相加，再仅在最后转换为 FP64。

审查另外使用 Python `Fraction(float(value))` 逐项构造精确有理数并求和，未调用 family 的任何参考函数。核验 **256 个候选 × 4 行，共 1,024 个精确行和，涉及 12,288 个存储输入值**；所有结果与保存的 `math.fsum` 参考和整数 dyadic 参考完全一致。最终约分后分子最多 29 bits，分母为二次幂，因此最终和能由 FP64 精确表示，最后转换没有引入参考误差。

最大输入为 `2^80`，即约 `1.21e24`；这里只有十二项，距 FP64 最大有限值约 `1.80e308` 极远。因此该固定构造不存在 `math.fsum` 大项中间溢出的问题。精确整数实现也不存在定宽整数溢出。

case_74 的真实误差小于阈值约三个数量级，case_75 为 100% 相对误差，两者均远离判定边界。公开代码只因 seed 不同而改变固定输入；初始 probe 的错误通过/拒绝没有被当作真值来源。随后保存的 GPU 冻结记录已确认编译后的 compensated kernel 与仿真一致，并保留真实 initial-probe 输出和输入不可变检查。

## case_74 r1 相同输入与判读范围的补充核验

在 `ea_methods_v3_r1` 完成后，另行核对了 case_74 solo 的全部 **5 份**原始 API 请求和 debate 的全部 **9 份**原始 API 请求。每份请求的 `Current Run State.artifact` 都包含完整冻结源码与完整合同：源码只增加行号、合同原文直接提供；没有截断。两组的这两项内容逐字一致，恢复后的 SHA256 与公开文件、`trace_meta.json` 和 `private_data/validation_gpu.json` 相同。Source-only 的实际 `request.json` 也包含相同的完整源码和合同。

公开 meta 为 `passed: null`、`status: unverified`，工具上下文相应为未知标签；保存的提示没有得到私有真值。Agent 执行镜像通过同一 runner 只挂载公开 `eval_cases`，没有挂载私有 family、answer key 或 GPU 真值文件。初始 probe 的实际输出和 `passed: false` 是同样提供给三组的待审查证据，并明确注明不是 oracle verdict。未发现本例有答案泄露、输入材料不对等或 harness 更换候选源码的迹象。此结论限定于已检查的文件、请求和镜像配置，不把文件名检索视为对任意外部知识来源的绝对证明。

Solo 的 [t7 原始 probe](../traces_glm/case_74/solo/r1/probes/t7_probe.py)把 `np.float64.sum(axis=1)` 的结果标为 exact，得到错误的零目标，进而误拒；实际 kernel 输出与冻结合格输出相同。Debate 的[最终判断](../traces_glm/case_74/debate/r1/verdict.json)使用实际执行的 `math.fsum` 参考得到正确 trust。其保存的 history 显示：Skeptic 一度接受零参考，Describer 后续明确指出该 FP64 参考有吸收误差，Experimenter 因而执行了精确求和实验。这提供了本次交叉复核纠正错误参考的具体过程证据。

这仍是在一个冻结工作负载上的一次成对差异。两组拥有相同 32,768 累计输出上限，但实际成本不相等：solo 使用 1,552 输出 tokens、API 估算 $0.018670；debate 使用 10,031 输出 tokens、API 估算 $0.089860。角色提示、调用数、输入 tokens 和实际计算量不同。因此可比较的是当前两条工作流，不能仅凭本例将改善完全归因于 agent 数量。若第二轮同例重复，只能加强该开发案例上的可重复性；推广到新案例仍需要新的冻结、平衡工作负载及保留全部结果的验证。
