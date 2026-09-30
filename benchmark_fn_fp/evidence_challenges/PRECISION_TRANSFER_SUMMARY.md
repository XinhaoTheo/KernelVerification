# 六个新输入的确认结果：case_76–case_81

2026-09-24。全部六例 × 三组的 18 次实验已完成；没有 API 传输失败、预算耗尽
或弃答。92 份原始请求/响应完整，自动冻结哈希、预算和 trace 检查均通过。
本批严格遵循[事前协议](PRECISION_TRANSFER_PROTOCOL.md)，原始结果见
[逐次报告](PRECISION_TRANSFER_REPORT.md)。

| 组别 | 正确 | 明确误判 | API 估算，不含 GPU |
|---|---:|---:|---:|
| 无工具单次调用 | 2/6 | 4 | $0.015287 |
| Solo | 5/6 | 1 | $0.128306 |
| Debate | 6/6 | 0 | $0.474371 |

合计 **$0.617964**。Debate 多答对一例；尚未达到预先要求的至少多两例的全组
第二轮触发条件，因此不追加付费复测或挑选局部重跑。当前结果是小幅的新输入
迁移信号，未满足“在新输入两轮重复优势”的门槛。

| Case | GPU 真值 | 实际相对误差 | 无工具 | Solo | Debate |
|---|---|---:|---|---|---|
| case_76 | trust | 7.975729e-8 | reject，错 | trust | trust |
| case_77 | reject | 0.885631 | trust，错 | reject | reject |
| case_78 | trust | 3.378736e-8 | trust | reject，错 | trust |
| case_79 | reject | 0.661099 | trust，错 | reject | reject |
| case_80 | trust | 5.725814e-8 | trust | trust | trust |
| case_81 | reject | 1.0 | trust，错 | reject | reject |

case_78 的 solo 把普通 FP64 reduction 的零结果误认成精确参考，置信度 0.99 地
拒绝正确 kernel；debate 用 `Fraction(float(v))` 建立精确目标并执行真实 GPU
kernel，核实误差为 `3.3787355e-8`，正确接受。这与开发例 case_74 的错误参考机制
一致。case_78 的 debate 在 Describer 阶段就指出参考不可靠；不能声称每次成功都
来自纠正另一角色已经给出的错误 verdict。

同样要保留反例：case_76、case_80 的 solo 自行选择精确参考，正确验证了两个合格输入。
所以该机制没有稳定地使 solo 失败。case_79 的 solo 虽然最终 reject 与标签一致，
却依赖同样错误的零参考并报告伪误差；它的正确标签不代表验证证据成立。
部分 debate 辅助模拟或编译器诊断也有错误，详见原始审查；最终主证据与辅助
解释质量分别记录，没有事后更改标签评分。

构造保持同一 4×12 FP32 补偿求和代码、容差与合同，仅换新的 seed。固定区间
`203600..204111` 的全部 512 个候选保留：9 个合格、401 个非零失败、102 个
零输出失败。按事前规则取三个合格、两个非零失败、一个零输出失败，未按模型
回答筛选。新范围与开发范围无重叠。精确整数 dyadic、math.fsum、另写的 Fraction
参考逐项一致；独立逐步 FP32 rounding 与原仿真也逐项一致。每个选中输入都
先通过实际 T4 十次重复验证，再附加真实初始观察并冻结。

资料：[独立 oracle 审查](PRECISION_TRANSFER_ORACLE_AUDIT.md)、
[case_76–case_78 证据审查](PRECISION_TRANSFER_AUDIT_E15_E17.md)、
[case_79–case_81 证据审查](PRECISION_TRANSFER_AUDIT_E18_E20.md)、
[所有候选](private_data/search_precision_transfer.json)。

这是一种自适应选出的机制内部的新 seed 检查；不是新算法家族，也不足以证明
一般多 agent 优势。输出上限相同，但实际调用次数、输入 tokens、费用和 GPU
时间不相同；这些差异保留在每次 trace 中。
