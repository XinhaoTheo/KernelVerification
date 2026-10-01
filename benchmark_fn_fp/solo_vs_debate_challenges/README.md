# Solo vs 多角色 debate：case_62–case_81

这 20 题测试：**面对一份真实但可能不充分的初始实验，能否检查参考是否可靠、输入域是否覆盖、观察时机是否符合合同**。三组得到相同 kernel、合同、生成器和初始 T4 输出。初始测试结果是待审查证据，不是真值。

现有结论：最初十题与连续盒域题均未区分 solo/debate。错误求和参考在 case_74 上两次触发 solo 错、debate 对，但 case_75 两次 debate 弃答抵消了总体正确率收益；六个新 seed 的确认仅多答对一题，未达到追加复测门槛。**有具体机制信号，尚未证明稳定的整体 debate 优势。**

看题从 [统一公开目录](../triton_eval_cases/)或[全局题目索引](../CASE_INDEX.md)进入；全部运行、失败与 probe 在 [GLM traces](../traces_glm/INDEX.md)。本目录只保留 README＋`private_data/`；程序在 [eval_scripts/solo_vs_debate_challenges](../eval_scripts/solo_vs_debate_challenges/)。新 CLI 名称为 `solo_vs_debate_challenges`，历史 trace 的 dataset `evidence_challenges` 保留用于追溯。

## 题型与合同

| 案例与合同 | 初始证据的盲区 | 独立真值/完整范围 |
|---|---|---|
| [62/63](../triton_eval_cases/case_62/problem.txt) | candidate 与 tester 共用不稳定的 FP32 原始矩协方差 | 独立居中回归预测误差 0.00206523 / 0.25584796，阈值0.05 |
| [64/65](../triton_eval_cases/case_64/problem.txt) | 缓存重排＋追加只测五个短操作字 | 全部121个长度0–4操作字；64全通过，65有12个失败，最大绝对误差7.625，阈值1e-5 |
| [66/67](../triton_eval_cases/case_66/problem.txt) | 单通道/两两正确不保证四阶分布 | 固定1024种子、8通道、70组四通道分布；66全部正确，67恰一组有依赖 |
| [68/69](../triton_eval_cases/case_68/problem.txt) | 每个view单独对齐不等于一个全局共享变换 | 穷举24×16=384变换；共同最小误差0 / 0.8316711，阈值0.05 |
| [70/71](../triton_eval_cases/case_70/problem.txt) | 候选和tester都保留可变状态视图 | 六步更新结束后观察原历史对象；误差0.00574470 / 0.89898037，阈值0.025；最终状态另需1e-5 |
| [72/73](../triton_eval_cases/case_72/problem.txt) | 41个smoke点不能证明整个连续盒域 | [-1,1]^12的凸残差证书；72全域上界0.7187501<1，73真实顶点反例1.34375 |
| [74/75](../triton_eval_cases/case_74/problem.txt) | 普通FP64求和也会丢失小项，不能自动当oracle | 对存储的4×12 FP32输入求精确实数和；误差1.92358e-8 / 1，阈值1e-5 |
| [76–81](../triton_eval_cases/case_76/problem.txt) | 同一求和参考机制的六个全新seed | 三个合格、两个非零错误输出、一个零错误输出；不是新增算法家族 |

具体输入、dtype、别名和允许修改的状态以各题合同为准。不能把所有别名都当失败，也不能把私有验证规定的完整域暗中缩成初始 smoke 样本。

## 各批结果

所有批次使用 Fireworks `accounts/fireworks/models/glm-5p3`、low reasoning，以及同一套无工具/solo/四角色debate提示。Source只有一次请求；solo最多十轮、debate四轮。三臂**每次运行共享累计32768输出tokens**，工具臂跨全部角色扣减，逐请求cap随剩余额度下降；SDK自动重试关闭，未另设最终verdict预留。输入tokens、总费用、GPU时间未匹配，不能称完全同成本。

当前rN仅为case/arm内存储序号，实验批次以 `original_trial` 识别。明确错误、弃答、预算耗尽、服务失败分开记；标签成绩与证据质量也分开，不因事后审查改判原成绩。

### 首批四题：case_62–case_65

`ea_pilot_r1` 的12个槽位全部完成：source 3/4，solo/debate各4/4，费用约$0.355581。60个API请求均完整，无传输失败、截断或弃答。两种工具系统都自行修正了参考/覆盖盲区；未触发原协议的第二轮及六题扩展。这里的阴性结果仍保留，当前20题还包含后来**另开版本**的16题，不能再说整个目录“只有四例”。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED PILOT RESULTS -->

生成时间：2026-10-01T04:01:50.797729+00:00

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ea_pilot_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 10 | solo | 4 | 4 | 0 | 0 | 0 | 0 | $0.079168 |
| ea_pilot_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 4 | debate | 4 | 4 | 0 | 0 | 0 | 0 | $0.264906 |
| ea_pilot_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 4 | 3 | 1 | 0 | 0 | 0 | $0.011507 |

共 12 次记录；API 费用估算 $0.355581。费用未知 0 次，费用记录不完整 0 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

指定窗口扩展门槛： **未达到**; 复验中重复纠错的题目： none; 配对明确错误净改善： 0.

纠正明确错误与解决弃答分别统计。达到数值门槛后仍需独立审核关键运行证据；该报告不会启动新实验。

<!-- END GENERATED PILOT RESULTS -->

</details>

### v2：case_66–case_71

`ea_methods_v2_r1` 共18槽：source 3/6（66弃答、67/70明确错误），solo/debate各6/6；费用约$0.593313。未触发第二轮。联合分布、共享变换、历史输出三种新机制都未产生solo错/debate对；case_67 debate用完整CPU结构分析，不能说每条工具trace都另跑GPU。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED METHODS V2 RESULTS -->

生成时间：2026-10-01T04:01:51.097478+00:00

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ea_methods_v2_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 10 | solo | 6 | 6 | 0 | 0 | 0 | 0 | $0.127509 |
| ea_methods_v2_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 4 | debate | 6 | 6 | 0 | 0 | 0 | 0 | $0.455780 |
| ea_methods_v2_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 6 | 3 | 2 | 1 | 0 | 0 | $0.010024 |

共 18 次记录；API 费用估算 $0.593313。费用未知 0 次，费用记录不完整 0 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

指定窗口扩展门槛： **未达到**; 复验中重复纠错的题目： none; 配对明确错误净改善： 0.

纠正明确错误与解决弃答分别统计。达到数值门槛后仍需独立审核关键运行证据；该报告不会启动新实验。

<!-- END GENERATED METHODS V2 RESULTS -->

</details>

### v3：case_72–case_75

`ea_methods_v3_r1/r2` 全四题三臂各两轮，共24槽：source 2/8（5明确错、1弃答），solo 6/8（2明确错），debate 6/8（2弃答）；费用约$0.962978。137个API调用完整，无传输失败或累计预算耗尽。Case_74同例纠错重复出现，满足开发机制的扩展门槛；case_75的两次弃答不改成模型曾尝试提交的reject，因此总体正确率仍打平。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED METHODS V3 RESULTS -->

生成时间：2026-10-01T04:01:51.428083+00:00

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ea_methods_v3_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 10 | solo | 4 | 3 | 1 | 0 | 0 | 0 | $0.118328 |
| ea_methods_v3_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 4 | debate | 4 | 3 | 0 | 1 | 0 | 0 | $0.273171 |
| ea_methods_v3_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 4 | 1 | 3 | 0 | 0 | 0 | $0.019717 |
| ea_methods_v3_r2 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 10 | solo | 4 | 3 | 1 | 0 | 0 | 0 | $0.084322 |
| ea_methods_v3_r2 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 4 | debate | 4 | 3 | 0 | 1 | 0 | 0 | $0.439262 |
| ea_methods_v3_r2 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 4 | 1 | 2 | 1 | 0 | 0 | $0.028178 |

共 24 次记录；API 费用估算 $0.962978。费用未知 0 次，费用记录不完整 0 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

指定窗口扩展门槛： **通过**; 复验中重复纠错的题目： case_74; 配对明确错误净改善： 2.

纠正明确错误与解决弃答分别统计。达到数值门槛后仍需独立审核关键运行证据；该报告不会启动新实验。

<!-- END GENERATED METHODS V3 RESULTS -->

</details>

### 新seed确认：case_76–case_81

`ea_precision_transfer_r1` 共18槽：source 2/6、solo5/6、debate6/6，唯一solo错/debate对为case_78；费用约$0.617964。92份API请求/响应完整，无传输失败、预算耗尽或弃答。Debate仅多答对一题，未达到事前规定的“至少多两题”全组第二轮门槛，已停止该批扩量，没有挑选个别题补跑。Case_76/80的solo自行找到精确参考，反驳“solo普遍不能解决”的解释。

<details>
<summary>查看逐批运行统计与失败记录</summary>

<!-- BEGIN GENERATED PRECISION TRANSFER RESULTS -->

生成时间：2026-10-01T04:01:51.753924+00:00；题目范围：case_76–case_81。

| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ea_precision_transfer_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 10 | solo | 6 | 5 | 1 | 0 | 0 | 0 | $0.128306 |
| ea_precision_transfer_r1 | glm-5p3 / fireworks | 每次 32768 / 总 32768 / 推理 low / 轮次 4 | debate | 6 | 6 | 0 | 0 | 0 | 0 | $0.474371 |
| ea_precision_transfer_r1 | glm-5p3 / fireworks | 每次 32768 / 总 unknown / 推理 low / 轮次 unknown | single_call | 6 | 2 | 4 | 0 | 0 | 0 | $0.015287 |

共 18 次记录；API 费用估算 $0.617964。费用未知 0 次，费用记录不完整 0 次。

“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。

逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) 和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。

新随机种子复验门槛： **未达到**; 复验中重复纠错的题目： none.

| 指定批次 | 无工具正确 | Solo 正确 | Debate 正确 | 每组固定分母 | Debate − solo | 全部结束 |
|---|---:|---:|---:|---:|---:|---|
| ea_precision_transfer_r1 | 2 | 5 | 6 | 6 | 1 | True |
| ea_precision_transfer_r2 | 0 | 0 | 0 | 6 | 0 | False |

弃答与失败保留在固定分母中。这是自适应选定机制内的新随机种子测试，不代表广泛泛化；数值门槛通过后仍需独立审核证据。

<!-- END GENERATED PRECISION TRANSFER RESULTS -->

</details>

<details>
<summary>各批事前门槛与构造规则</summary>

四阶段均于各自首次模型调用前记录设置，已评测题不改源码、合同、阈值或标签。公共 `initial_probe` 的实际T4输出在preview后、冻结前附入合同，不能伪造；三组输入相同，私有family、oracle和搜索记录不进入agent镜像。独立CPU参考与真实T4十次决定性测量先于模型评测，保留全部候选与失败验证记录。

- **首批**：先四题×三臂一轮；只有明确solo错/debate对才全四题再测一轮。相同case重复纠错、有效证据、正向差异多于反向明确错误，才按冻结规则新增六个平衡题并全组三臂测两轮。弃答、耗尽和provider失败不触发。该原计划未进入第二轮，后续v2不是伪装成成功扩量。
- **v2/v3**：同样先完成全组，再由明确solo错/debate对触发全组第二轮；同例重复、正向多于反向明确错误、独立证据审查通过才继续新seed。此门槛统计明确纠错，不等于总正确率提高；v3的case_75弃答必须另计。
- **新seed确认**：保持同一4×12生成器、Neumaier实现、精确和合同和1e-5阈值，预先固定新范围203600–204111。按seed升序取三个误差≤半阈值的合格、两个非零失败、一个零输出失败，失败均>两倍阈值，不依模型答案选例。第一轮完成全部18槽后，仅当debate至少多对两题且有一个明确纠错才全组再测。重复优势还需两轮均领先、至少同一题纠错重复及有效证据；本次没有达到第二轮触发条件。

公共材料、模型、role prompts和工具权限在这些批次中未同时改造。相同输出额度不隔离角色数量、输入上下文或计算量的影响；失败槽位不由成功重试替换，工具内部修复仍属于原trial。所有72次试验共378个实际API请求，无已知缺失usage；有真实probe/ledger错误，不能将“无API失败”写成“全部操作无错”。

</details>

<details>
<summary>独立真值、搜索范围与边界记录</summary>

全部私有证据位于 [private_data](private_data/)，最终真值以 [validation_gpu.json](private_data/validation_gpu.json)为准；`validation_attempts/`保留13份preview、freeze和失败记录。环境为T4、PyTorch2.8.0、Triton3.4.0、NumPy1.26.4。`oracle_source_sha256`记录原GPU冻结时的验证源码；这次只整理路径，没有重新运行GPU或产生新实验。

| 机制 | 独立核验与选例范围 |
|---|---|
| 回归参考 | FP64居中增广最小二乘＋Decimal80；[256 seeds](private_data/search_reference.json)中166合格，取最小/最大误差。 |
| 状态操作 | FP64正向逻辑状态＋整数反向祖先；[64×121](private_data/search_sequence.json)全部检查，17seed全合格、47有违反，按顺序选正反例。每个操作字从新鲜状态开始。 |
| 联合分布 | 1024seed完整直方图＋GF(2)仿射秩，全部70个四元组；[256候选](private_data/search_joint.json)。四阶以下正确不能证明四阶正确。 |
| 共享变换 | 全384变换＋独立有符号赋值代价，保留[256候选](private_data/search_alignment.json)；局部解非唯一，必须检查共同解，不能只比较局部首选。 |
| 历史对象 | [64候选](private_data/search_mutation.json)和独立精确参考；候选返回对象保持原别名直到六步结束，参考保存各步独立值；分别检查允许修改的状态和禁止修改的梯度。 |
| 连续盒域 | 正系数ReLU残差非负且凸，最大值可取4096顶点；独立64个active-subset支持函数以整数/Fraction复核全部[64候选](private_data/search_box_relu.json)。base=0.25x0+0.5x1在全域另加1e-7舍入/FTZ界，不能说内部点base总是精确。未选中seed194063的M=1只表明上界无法证明通过，已改为requires_boundary_analysis；578个邻近FP32点的补查也不构成全域证明。 |
| 精确求和 | [256开发候选](private_data/search_precision_oracle.json)的1024行，用math.fsum、精确dyadic整数和另写Fraction逐项一致。±2^80、±2^30消去后八个正小量必须保留；普通顺序FP64与FP32 compensation本身都可能丢失小项。 |
| 新seed | [512完整候选](private_data/search_precision_transfer.json)与193600–193855开发范围不交叠；9合格、401非零失败、102零输出失败。另写Fraction核验2048行、struct逐步FP32舍入复现全部候选；最后FP64参考转换精确。选中seed为203635/203600/203795/203601/203820/203604，对应76–81。 |

所有公开generator重建的输入匹配私有hash；已选box/precision题远离阈值，未选边界记录的修正不改其标签。Case_77 CPU/GPU最终归一化差2.22e-16，不是输出或参考不一致。输入材料公平性另逐份核对case_74 r1的5份solo、9份debate请求及source prompt，完整源码/合同相同且无截断，没有私有答案。

</details>

<details>
<summary>决定性证据、真实失败及未被纠正的解释</summary>

这里区分真实GPU执行、对共同初始GPU观测的CPU复核、静态/结构证明，以及只有假设的说明。标签正确不自动意味着参考正确；私有oracle或源码支持的属性也不能冒称agent本次测量过。完整文件和失败stderr保留在各trace。

| 范围 | 关键证据与局限 |
|---|---|
| 62/63 | [solo t7](../traces_glm/case_63/solo/r1/probes/t7_probe.py)和debate均以独立居中FP64参考测量真实GPU。62 debate t13把FP32统计转Python float后用混合精度，不能叫精确复现initial_probe；其c2“旧零误差不充分”也不能因新实验通过就被驳回。62/63的辅助向量归约不等于候选逐元素顺序，2.24%协方差归因不是实际kernel测量。Source62虽对却依赖共用错误参考；64误拒，65对但提出了无效辅助反例。 |
| 64/65 | [64 solo](../traces_glm/case_64/solo/r1/probes/t7_probe.py)独立GPU穷举121字，逐项检查shape/dtype/finite。Debate64 t12/t13参考broadcast失败后修复；t15只统计超阈值条目，不能把“无超标”说成主动测得全域最大误差0，且没做finite检查。65合法反例(0,1,2)误差3.8125、(0,1,2,2)误差7.625，solo/debate均找到；非identity不一定失败，非involution才是方向问题。 |
| 66/67 | [66 debate](../traces_glm/case_66/debate/r1/transcript.md)实际GPU计数支持trust，却把顺序XOR-fold错当对原始值的一次性移位，编造8×5低秩模型；仿射offset不能增加秩，末轮review未纠正。其list(None)、combinations索引报错后修复。67 solo辅助formula_match也写错fold，但主GPU四阶计数有效；[67 debate](../traces_glm/case_67/debate/r1/transcript.md)是完整CPU整数仿真＋GF(2)证明，没有本次GPU执行。找到的确是四通道依赖；“任何非空依赖都破坏四阶”过强。 |
| 68/69 | [68 solo](../traces_glm/case_68/solo/r1/probes/t7_probe.py)及其余主probe穷举384共享变换，局部首选不同不等于无共同解。69 solo t6打印NameError后t7重跑成功。69 debate只检features未改，不是全部输入检查；注释的NaN sentinel未实际设置，输出转FP64后也不能证明原dtype。模型probe未保存完整输出hash，只核对合同误差与最优变换。 |
| 70/71 | [70 solo](../traces_glm/case_70/solo/r1/probes/t7_probe.py)与debate都在完整更新后才stack候选原对象，同时参考每步copy。71 solo首次混用CPU/CUDA比较失败后修复；torch.equal(g, regenerated_g)不是独立uint8字节检查。70的别名仍在容差内，不能因“存在别名”误拒。 |
| 72/73 | 两轮两工具臂均建立顶点证书或有效GPU反例；[72 solo r1 t11](../traces_glm/case_72/solo/r1/probes/t11_probe.py)在执行后才clone，输入未修改检查无效；73 debate r2该字段硬编码true。缺base舍入界、把逐神经元最大值和3.3125当实际联合最大值、用零分母floor形成巨大smoke倍率，均属论证限制。广播/设备/矩阵乘标量错误有后续修复；被ledger拦下的probe未执行。72 r2的20万随机点仅CPU搜索，不能代替4096顶点证书，GPU batch仍合法。 |
| 74 | [solo r1](../traces_glm/case_74/solo/r1/probes/t7_probe.py)/r2均把np.float64.sum的0当exact，以0.99置信误拒，非执行故障。[debate r1](../traces_glm/case_74/debate/r1/transcript.md)有Skeptic接受错误参考→Describer指出吸收→Experimenter用fsum＋真实GPU→Judge纠正的链条，E=1.92357928e-8。r2修复Fraction(np.float32)错误后用Fraction复核共享初始T4输出，没有重新执行GPU。 |
| 75 | Solo两轮用fsum＋真实GPU得到E=1，r2缺NumPy import后修复。[debate r1](../traces_glm/case_75/debate/r1/transcript.md)发现参考错误但承诺的补测未执行；[r2](../traces_glm/case_75/debate/r2/transcript.md)以AST等价Triton副本＋Fraction/fsum完成有效E=1测量，却未finalize决定性claim。两次reject请求被ledger拒绝，最终弃答，不可追溯重评分；两次均未耗尽预算。 |
| 76–78 | [78 solo](../traces_glm/case_78/solo/r1/probes/t7_probe.py)再次误用FP64零参考，以0.99置信误拒；[debate t13](../traces_glm/case_78/debate/r1/probes/t13_probe.py)用Fraction＋原GPU得到E=3.3787355e-8，但Describer起初已识别参考问题，并非每次都是角色间纠正已出错verdict。其t12 full模拟重复计首个2^80，未修复，不影响t13主证据。76 solo和77 solo自行用Fraction；76 solo、77 debate有Fraction(np.float32)失败后以float精确转换修复。77 debate跑的是逐AST验证等价源码副本，不是修正kernel。 |
| 79–81 | [79 solo](../traces_glm/case_79/solo/r1/probes/t7_probe.py)标签碰巧对，仍用错误零参考和伪误差；debate用Fraction＋GPU得到0.66109915，其PTX零opcode计数未存文本，不能证明无编译器问题。[80 solo](../traces_glm/case_80/solo/r1/probes/t7_probe.py)自己找到精确参考，Python dict stdout仍是有效原始数值而非解析JSON。81 solo修复Fraction类型错误；[debate t17](../traces_glm/case_81/debate/r1/probes/t17_probe.py)真修复参考得E=1，但未finalize c1，最终依c2机制和raw t17拒绝，登记缺口仍保留。 |

此外，不少claim缺 `scope_rationale`、description参数不合格式后在原运行内修复。大数间距的解释也需准确：FP64在2^80附近间距为2^28，±2^30可精确加入，丢失的是单位量级小项；“补偿大项相邻”仍不够，第一大项之前的小量也可能被correction抹掉。Case_76/80 solo成功与case_79偶然标签正确都必须保留，不按证据质量事后重算二元分数。

</details>

## 使用与未验证的下一步

离线重建四个结果块，不产生模型调用：

```sh
python benchmark_fn_fp/eval_scripts/solo_vs_debate_challenges/report.py
python benchmark_fn_fp/eval_scripts/solo_vs_debate_challenges/report_methods_v2.py
python benchmark_fn_fp/eval_scripts/solo_vs_debate_challenges/report_methods_v3.py
python benchmark_fn_fp/eval_scripts/solo_vs_debate_challenges/report_precision_transfer.py
```

`--json`可选导出`private_data/reports/`，原始记录仍在统一trace。真实GPU验证程序为 `eval_scripts/solo_vs_debate_challenges/validate_modal.py`，preview/附加真实初始观察/freeze用于构造，不能重写已评测合同。新模型运行使用共享runner的 `--dataset solo_vs_debate_challenges --cases ...`，另有API/GPU费用。

Case_75暴露了具体编排缺口：实验请求通常由open claims驱动，改成inconclusive后承诺的修复可能失去调度；修复成功也可能没登记。未来可在原预算内增加带claim ID的有界修复请求、补测与显式finalize检查，但这是**未测试的下一版本建议**，不能算本批改进或重评分理由。仍需保留无法修复时的弃答，不因probe成功就自动confirm。

这是一条自适应开发轨迹。新seed确认只是在已选机制内部迁移，不是新算法家族，也不是一般多agent优势证明。同例重复不是独立新题；未来需固定构造、完整保留负例，以同预算solo自审、独立多次solo汇总等对照区分复核流程、采样和角色交互的作用。Confidence仅原样记录，未校准为可靠概率。
