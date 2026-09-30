# case_50–case_61：六组固定输入数值验证案例

case_50–case_61 新增六组、十二例。每组两个案例使用同一个计算方法、同一份数值合同，公开源码仅输入生成器的 PCG64 seed 不同；一例满足合同（`trust`），另一例不满足（`reject`）。选择依据是数值计算，未使用模型回答筛选 seed。

这些案例测试的是：看到一个存在数值风险的近似实现后，能否判断它在**这一份固定输入**上是否越过误差阈值。它们没有要求实现对任意输入都成立。合同使用指定分母下的相对 L2 误差，并非逐元素 `allclose`；中间量误差大，也不能自动推出最终输出不合格。

实验安排见 [OZ_PROTOCOL.md](OZ_PROTOCOL.md)，模型判断、失败记录、费用及复验结果见持续更新的 [OZ_REPORT.md](OZ_REPORT.md)。本文件只说明构造与数值真值，不把构造成功当作模型实验成功。

| 组别与公开内核 | 机制 | 最终 T4 误差，按案例顺序 | 容差 | 冻结标签，按案例顺序 | 合同 |
|---|---|---:|---:|---|---|
| [O](eval_cases/case_50/kernel.py) / [P](eval_cases/case_51/kernel.py) | 近共线向量正交投影，再归一化小残差 | 0.154365% / 6.045526% | 1% | trust / reject | [投影合同](eval_cases/case_50/problem.txt) |
| [Q](eval_cases/case_52/kernel.py) / [R](eval_cases/case_53/kernel.py) | 坐标量化改变最近邻路由，随后提取 embedding | 0% / 134.018400% | 10% | trust / reject | [路由合同](eval_cases/case_52/problem.txt) |
| [S](eval_cases/case_54/kernel.py) / [T](eval_cases/case_55/kernel.py) | 振荡函数的固定 32 点中点积分 | 0.155252% / 11.888437% | 3.5% | trust / reject | [积分合同](eval_cases/case_54/problem.txt) |
| [U](eval_cases/case_56/kernel.py) / [V](eval_cases/case_57/kernel.py) | Fourier 重建只保留 16 个频率中的前 6 个 | 7.170936% / 23.046427% | 15% | trust / reject | [频率截断合同](eval_cases/case_56/problem.txt) |
| [W](eval_cases/case_58/kernel.py) / [X](eval_cases/case_59/kernel.py) | 病态 SPD 矩阵的 FP32 消元与 log-determinant | 0.0000202355% / 0.0711920% | 0.01% | trust / reject | [logdet 合同](eval_cases/case_58/problem.txt) |
| [Y](eval_cases/case_60/kernel.py) / [Z](eval_cases/case_61/kernel.py) | 平方距离展开式相消，经 RBF 权重传播到归一化预测 | 0.0142222% / 29.240636% | 5% | trust / reject | [RBF 合同](eval_cases/case_60/problem.txt) |

表中百分比等于各自合同误差值乘以 100，使用每例十次实际 T4 执行的最大误差。十二例均通过输出形状、有限值、输入未被修改及重复执行检查，每例十次输出逐位相同。环境为 Tesla T4、Torch 2.8.0+cu128、Triton 3.4.0、NumPy 1.26.4、Python 3.11.12。精确误差和源码、输入、输出 hash 保存在 [private_data/validation_gpu.json](private_data/validation_gpu.json)，表中小数作了显示舍入。

## 为什么仅有误差量级估计仍不足以确定标签

**case_50/case_51：归一化会放大方向误差。** 输入的 `b` 接近 `1.125*u`。FP32 投影系数的微小偏差会改变残差方向，最终还要除以很小的残差范数。知道“有严重相消”只能定位风险；是否超过 1% 取决于这一组向量的逐步舍入。工具可以执行内核，并用 FP64 重新中心化公式计算参考；独立 Decimal80 参考直接计算原始未中心化公式，避免两个参考复用同一条相消路径。实现与参考见 [orthogonal_projection.py](families/orthogonal_projection.py)。

**case_52/case_53：连续距离误差会触发离散选择变化。** 16 个候选到 query 的真实距离接近，坐标先量化到 1/8 网格再比较。量化可能改变最近邻，也可能保留同一个最近邻；选中同一行时最终 embedding 完全正确。工具可以计算未量化的 FP64 距离和量化距离，应用相同的最小索引 tie-break，再比较最终 embedding，而不必从量化误差大小猜测路由。独立 Decimal80 距离核验获胜索引，见 [quantized_routing.py](families/quantized_routing.py)。

**case_54/case_55：采样混叠的实际贡献依赖频率、幅度和相位。** 32 点网格不足以普遍精确积分输入中的所有高频项，但某个固定输入的误差仍可能相互抵消，或贡献很小。工具可以从公开 FP32 系数计算连续积分的解析式，与实际 Triton 中点积分输出比较；独立 Decimal80 三角函数级数核验解析参考。不能把 32 点离散和本身当作参考，因为合同要求连续积分。见 [quadrature.py](families/quadrature.py)。

**case_56/case_57：截断是否合格取决于被省略部分占总能量的比例。** 两例都省略相同的十个频率，但实际系数能量不同。工具既可用完整频谱的 FP64 inverse FFT 与实际内核比较，也可用 Parseval 关系计算理想截断误差作为诊断；独立参考是直接三角展开配合 `math.fsum`。Parseval 提供可计算的解析关系，并不意味着只能运行 GPU 才能判断，但需要知道这份 seed 生成的实际系数。见 [spectral_filter.py](families/spectral_filter.py)。

**case_58/case_59：相近的条件数量级不决定实际 logdet 误差。** 两例均为整数 Gram 矩阵加 `1/1024 * I`，参考矩阵严格正定。FP32 Schur 补更新中的舍入会改变较小的消元主元，再影响累计对数。最坏情况误差上界超过阈值，并不能证明这一个矩阵失败。工具可以对实际 FP32 矩阵计算 FP64 `slogdet`，再与真实内核比较；独立 Decimal80 消元计算 determinant 后取自然对数。见 [logdet.py](families/logdet.py)。

**case_60/case_61：距离误差还要经过指数、权重归一化和带符号求和。** 实现用 `||a||² + ||q||² - 2*a·q` 计算近邻距离，较大的公共偏移导致相消。随后计算 `exp(-16*d)`，再用带正负值的 targets 求归一化预测。某些 seed 的终值误差相消，另一些会放大；仅验证中间距离无法判定合同。工具参考应从 FP64 直接差分平方开始，完成全部 RBF 预测，再比较最终输出。独立 Decimal80 参考包含距离、指数和归一化，见 [distance.py](families/distance.py)。

上述机制说明粗略估计为什么可能不够，并不声称无工具模型在逻辑上不可能求解。严格解析推导、足够紧的误差界，或充分展开的计算同样可能确定标签。是否实际拉开差距，以预先固定的三组实验及重复结果为准。

## 已保存的 CPU 搜索范围与数值合格率

以下逐项统计六份 `search_log` 的全部 `calibration_candidates` 和 `candidates`。合格的定义仅为该候选的 **CPU 数值误差 <= 本族容差**；这里没有任何 LLM 判断，比例也不是无工具、solo 或 debate 的准确率。校准行与正式搜索行分开统计，不把挑出的十二例当作随机样本。

| 组别与完整日志 | 已保存校准：合格数 / 候选数 | 校准合格率 | 正式：合格数 / 候选数 | 正式合格率 | 正式 seed 范围（含两端） |
|---|---:|---:|---:|---:|---|
| [case_50/case_51](private_data/search_log_orthogonal_projection.json) | 33 / 64 | 51.56% | 121 / 256 | 47.27% | 830100–830355 |
| [case_52/case_53](private_data/search_log_quantized_routing.json) | 无单独校准列表 | — | 21 / 256 | 8.20% | 840100–840355 |
| [case_54/case_55](private_data/search_log_quadrature.json) | 23 / 32 | 71.88% | 193 / 256 | 75.39% | 711000–711255 |
| [case_56/case_57](private_data/search_log_spectral_filter.json) | 17 / 32 | 53.13% | 126 / 256 | 49.22% | 811000–811255 |
| [case_58/case_59](private_data/search_log_logdet.json) | 38 / 100 | 38.00% | 135 / 256 | 52.73% | 98200–98455 |
| [case_60/case_61](private_data/search_log_distance.json) | 139 / 200 | 69.50% | 192 / 256 | 75.00% | 119100–119355 |
| **保存行数合计** | **250 / 428** | **58.41%** | **788 / 1536** | **51.30%** | 六个固定范围 |

校准 seed 范围分别是 case_50/case_51 的 730100–730163、case_54/case_55 的 710000–710031、case_56/case_57 的 810000–810031、case_58/case_59 的 98100–98199、case_60/case_61 的 118100–118299。case_58/case_59 日志保留的是最终 `1/1024` regularizer 下的校准列表；case_60/case_61 保留的是最终 offset=16 下的校准列表。更早的 regularizer=1/64、1/256 和 offset=8、32 探索未逐候选保存在这些日志中；原有工具输出中的统计摘要已注明时间[补归档](OZ_EARLY_PARAMETER_EXPLORATION.md)。上表不冒充全部参数探索的完整记录，补归档也未恢复缺失的逐候选数据。

case_52/case_53 正式表包含过滤前的全部 256 个候选。选择时另要求真实前两名距离差 `>1e-5`、量化前两名距离差 `>=1/64`，以避免临界 tie 导致不稳定；210 个候选符合此筛选，其中 19 个数值合格。最终取按 seed 顺序遇到的第一个满足安全余量的合格例与不合格例。其余五组直接选各自正式 256 个候选中的最小、最大误差。

最终选中 seed 为 case_50/case_51=830230/830228、case_52/case_53=840104/840101、case_54/case_55=711165/711076、case_56/case_57=811164/811158、case_58/case_59=98339/98262、case_60/case_61=119130/119263。所有选择都要求合格例误差不超过容差的 75%、不合格例不低于容差的 125%，然后再以实际 T4 执行确认标签。阈值均在正式 seed 搜索和模型实验之前固定；case_50/case_51、case_58/case_59、case_60/case_61 使用前期数值校准确定阈值，case_52/case_53、case_54/case_55、case_56/case_57 的日志记录阈值在其对应搜索或校准前已固定。

## 工具证据与结论边界

一个可审核的工具判断应从公开 `make_inputs()` 重建实际 FP32 输入，调用该案例原有的 `run()` 执行真实 Triton 内核，独立实现合同的高精度参考，并保存输出形状、有限性、误差值和阈值比较。单纯写一段看似合理的测试代码、只测别的输入，或仅模拟中间算子，都不足以证明固定案例合格。实际模型是否做到这些，需要检查对应 traces，不能只从它最终答对推出它进行了有效验证。

构造用的 oracle、answer key 和搜索日志不挂载到模型的工具环境；参考必须由模型根据公开合同实现。构造复核记录见 [case_50–case_53 审查](OZ_CONSTRUCTION_AUDIT.md) 与 [case_54–case_61 审查](OZ_CONSTRUCTION_AUDIT_SZ.md)。源码与输入在第一次模型实验前冻结，实验之后不按模型反馈修改。

十二例是有意选出的、标签均衡的合成固定工作负载，而不是未参与开发的 held-out 测试集。选择最小/最大误差和少数合格路由会改变原始候选分布；例如 case_52/case_53 原始正式候选只有 8.20% 数值合格，但最终一合格一不合格。因此模型在这十二例上的表现只能证明这些选定机制上的可复验现象，不能直接估计一般 kernel 验证准确率，或证明所有无工具模型无法完成任务。

[预注册实验安排](OZ_PROTOCOL.md) 将所有十二例纳入相同 GLM、low 推理、每次 32K 上限的三次无工具和各两次工具复验，另为全部十二例保留一次默认推理 64K 对照。工具组允许多轮与多次模型调用，总计算量和费用并未匹配；工具优势也不等于 debate 优于 solo。最终比较必须同时保留明确误判、无结果、服务错误、反例与全部费用口径，详见 [实验报告](OZ_REPORT.md)。
