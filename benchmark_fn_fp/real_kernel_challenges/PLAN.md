# 真实 kernel 小幅修改与三组验证计划

2026-09-27。状态：方案与来源初筛；尚未构造新案例、运行 GPU 或调用评测模型。

目标是在真实 kernel 的完整验证任务中，测量独立复核能否减少 solo 的遗漏。
沿用无工具单次调用、solo＋工具、debate＋工具三组，GLM 经 Fireworks，原始
traces 统一记录到 traces_glm。历史实验 case_62–case_81 的数据和评分保持原样。

## 1. 来源池：先找五类，首轮只用两类

| 优先级 | 真实来源 | 要审查的义务 | 可考虑的小改动／历史修复 | 目前状态 |
|---|---|---|---|---|
| 首轮方向 1 | Liger RMSNorm；FLA fused residual LayerNorm 作替补 | forward、dX、dWeight、残差梯度；行式与块式 backward 路径 | 首选重放 dWeight 累加精度的真实修复；或明确标注的块合并／尾块 mutation | 找到 Liger PR #950；当前 T4 环境尚未复现 |
| 首轮方向 2 | vLLM Triton unified/paged attention | 调度路径、分页索引、Q/K 长度关系、GQA、mask 与数值输出 | 单个合法优化分支中的偏移或掩码适用条件；保留完整 dispatch | 已确认公开源码；具体历史修复需定位，否则标 synthetic mutation |
| 扩展候选 | Liger fused linear cross entropy | loss、dInput、dWeight、reduction、ignore_index、上游梯度 | 分块梯度合并／归一化／逐 token 权重应用位置的小修改 | 已确认源码；具体缺陷与触发范围待核验 |
| 扩展候选 | AutoGPTQ Triton quantized matmul | 解包、g_idx、scale 映射、合法 group/布局组合 | 一个量化分组映射或布局分支的最小修改 | 已确认源码；须避免重复已有简单 floor/ceil 案例 |
| 扩展候选 | Triton grouped GEMM | 多矩阵任务调度、各组 leading dimension、tile 边界 | 一个 group/tile 偏移或调度边界修改 | 已确认官方实例；只保留硬件支持且合同允许的配置 |

来源：

- [Liger RMSNorm](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py)
- [Liger 历史修复 #950](https://github.com/linkedin/Liger-Kernel/pull/950)：已合并，涉及 Llama 模式块式 backward 的 dWeight 累加精度；上游记录的测试硬件是 A100。该事实不等于已在本项目复现。
- [FLA LayerNorm](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm.py)
- [vLLM Triton attention](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py)
- [Liger fused linear CE](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- [AutoGPTQ Triton kernels](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/triton_utils/kernels.py)
- [Triton grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)

执行时锁定不可变 commit、许可证、依赖与完整源文件哈希。来源池的 main 链接仅
用于初筛，不作为可复现实验版本。记录每个被排除的来源及原因。以 T4 可运行为
资格条件；硬件／依赖不兼容不算 kernel 错误。若需更换 dtype、算法或硬件路径，
明确记录为适配或合成变体，不能把它写成原始历史 bug 的原样复现。

## 2. 如何“稍微修改”

每个家族准备一个经独立验证的合格版本和一个确有合同违反的版本。优先使用
历史 bug 的 pre-fix/fix 快照；找不到适合的历史缺陷时，采用明确标注的 synthetic
mutation。模型评估前固定标签与选择规则，保留全部构造尝试。

合成修改目标是一个局部机制，通常几行：归约精度、跨块合并、地址映射或优化
分支的适用条件。保留相关 host wrapper、launch、dispatch 和 backward 上下文。
独立记录依赖裁剪／兼容适配 diff 与真正改变语义的 mutation diff，不把两个
类型的改动混写成“只改一行”。

每对要求：同一数学合同与同一工作负载；至少一个共同的正常输入可通过；错误
版本存在稳定、合法的反例；修复版本在整个声明的有限测试域内通过。数值阈值
由算子、dtype 与既有测试契约确定，不能看到模型结果后移动。原生正确优化即使
看起来可疑，也保留为合格对照。

公开样本使用中性、打乱的 ID；不提供 patch 标记、bug/fixed 名称、标签、修复
说明或答案 manifest。保留应有的许可证与版权。禁止 seed 特判、隐藏输入范围、
错误参考和暗示答案的注释。配对版本不在同一模型会话展示。

## 3. Pilot 的四例与干净真值

四例 = 归一化正确／错误一对 + attention 正确／错误一对。不能仅因为无工具
或 solo 很快答对就删掉案例。后续新的设计版本应保留此前失败结果和费用。

归一化合同覆盖前向与声明的全部梯度；工作负载矩阵取固定、少量、合法的 shape、
layout 和 dispatch 组合。参考采用独立 FP64 数学实现＋autograd，另以解析梯度
交叉验证；小规模方向导数检查仅作辅助，不单独决定低精度 kernel 的标签。

Attention 使用显式解页、head 映射与 causal 坐标的 dense 参考，再以独立小规模
实现核对。关闭无关随机功能或预先固定其合同。所有测试组合必须处于被选版本
声明支持的范围。只对冻结的有限域给通过结论，不把有限抽样写成全输入域证明。

模型调用前：CPU 独立参考一致性检查 → T4 实测 → 决定性反例及主要边界十次重复
→ 检查全部配置、输出／梯度、shape、dtype、finite 和声明的输入不变性 → 冻结
全部公共依赖哈希、环境、真值与数值余量。候选原样运行，不能修好后代替原候选
验证。没有通过资格检查的案例不进入模型付费阶段。

三组都得到完整候选、完整合同、正确参考实现及相同公共生成器；均不提供私有
标签或决定性验证结果。工具组可以执行这些材料，无工具组只能阅读。

## 4. 先处理现有 harness 的公平性问题

本次只读检查发现：工具角色对带行号的 kernel/test 会截断到 12,000 字符，而
无工具 runner 读取完整 kernel.py/problem.txt，却不会自动读取独立 reference.py。

实施共同的 public bundle loader，给三组注入相同完整材料，冻结并核验实际
渲染文本与每个 helper 的哈希。支付前检查无截断、无缺依赖、无私有标签泄露；
超出模型上下文的候选需要等价依赖裁剪或退出，不允许静默截断。

case_75 已暴露的补测调度／证据登记缺口先作为独立 runtime 修复完成回归测试，再
冻结新 runner 版本。共享 ledger／调度变更检查 solo 兼容性；不放松证据要求。
新实验明确标注 runtime 版本，不用新流程重评分旧 traces，也不在首轮结果出来
之后只替 debate 改提示、加回合或修状态。

三组使用同一 GLM 模型和 reasoning 配置。Pilot 保留每次累计输出 32K 上限，
debate 所有角色共享；如实记录实际输入、输出、缓存、GPU probes/秒数与费用。
这只是相同输出上限，不称同成本。若出现收益，在独立成本确认中给工具两臂
相同 API 预算和 GPU 执行额度，并允许 solo 在其额度内继续自检。

## 5. 实验阶段与停止条件

| 阶段 | 规模 | 决策 |
|---|---|---|
| 来源和真值资格检查 | 五类来源池，首轮四例 | 无模型调用；先保证真实性、兼容性与标签 |
| Pilot r1 | 四例 × 三组 = 12 次 | 全部结束后，只有 debate 总正确数高于 solo 且至少一个有效的明确 solo 错／debate 对，才进入 r2 |
| Pilot r2 | 同四例 × 三组 = 12 次 | 两轮均有正确率净收益、同一案例明确纠错重复、主证据有效，才扩展 |
| 新案例确认 r1 | 新增三对，共六例 × 三组 = 18 次 | 使用新源／新缺陷机制，不只换 seed；完整记录负结果。若 debate 多答对至少两例且有有效明确纠错，进入第二轮 |
| 新案例确认 r2 | 同六例 × 三组 = 18 次 | 要求两轮均有净收益及同例明确纠错重复；否则报告不稳定或无收益 |

这一探索计划最多十个不同案例、60 次模型试验；不存在无限搜索直到拿到想要
结果的循环。任一门槛不满足，就停止该阶段的付费扩量、保留全部结果，再分析
机制；不能只重跑获胜案例或用重试替换原槽位。没有完成实验前不报告成功。

弃答、预算耗尽、执行／API 失败均计入固定分母中的未答对，并分别列出；只有
可审计的明确错误判断才构成推理纠错。正确标签与有效证据分别统计，避免重演
case_79 那样依赖错误参考却碰巧答对的情况。

若要声称额外角色比更多计算更划算，需要另行预先登记成本确认；不把它暗中
计入上述 60 次计划，也不凭输出上限相同就宣称已经排除了计算量因素。

## 6. 交付与记录

新数据集目录 real_kernel_challenges；新增案例使用 case_map.json 的 next_case_number
连续分配全局 case 数字编号（当前下一号为 case_106），实际配对关系
留私有 manifest，顺序与标签不建立可利用规律。公共文件包含 kernel、contract、
reference、input generator 与所有必要 helper；私有文件包含来源 commit／许可、
历史或合成标记、两个版本的 diff、候选搜索、独立真值及 GPU 验证。

traces_glm/<case>/<arm>/<trial>/ 保留完整 request、response、usage、工具代码、
stdout/stderr、错误、覆盖配置、证据与最终判定。费用未知时明确记录未知。

汇总表列出：case、来源、机制、历史/合成、真值、三组正确/误判/弃答、可复现
反例或验证证据、测试覆盖、输入输出 tokens、API 估算、GPU 时间与运行器版本。
报告整体结果及所有成本，并附逐例 paired comparison 和角色实际纠错链。

成功标准是新真实案例上可重复的净收益，不预设独立复核必胜；即使十例都有
差距，也只是初步工程证据，不能把开发筛选集直接当作一般能力的无偏估计。
