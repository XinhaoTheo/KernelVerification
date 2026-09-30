# 无工具 r1 内容审查

四条 source-only 调用均正常结束，有最终 verdict，完整 request/response/reasoning
和 usage；没有重试或以新结果替换原槽。此处讨论理由质量，不改按冻结真值计算的
二分类分数（3/4）。

| 案例 | 判定 | 与真值关系 | 理由的证据范围 |
|---|---|---|---|
| case_62 | trust，confidence 0.95 | 正确 | 依赖初始 FP32 reference 的零误差，并称其为 exact centered ridge；没有建立合同要求的高精度参考，标签正确不代表证据充分 |
| case_63 | reject，confidence 0.70 | 正确 | 识别共享原始矩计算的循环验证问题，但仅估计误差；实际合同误差由私有 T4 oracle 确认为 0.25584796 |
| case_64 | reject，confidence 0.90 | 错误 | 声称任意重排后追加都会错，忽略当前 seed 的置换组合性质；真实 121 个合法序列全部精确通过 |
| case_65 | reject，confidence 0.85 | 正确 | 正确指出 order 必须等于其逆的条件；但把 (0,1,0) 也列为潜在非对合组合，这项辅助说明不成立；真实反例包括 (0,1,2) |

case_64 的错误不能归因于秘密输入：PCG64 seed、生成器、操作域均公开。无工具模型
没有实际求出置换并检查组合，因此把普遍可疑的源码当成了当前固定数据必然失败。
case_62 的 confidence 0.95 同样不能解释为经校准的概率；本实验没有重标定 confidence。

原始结果：[case_62](../traces_glm/case_62/single_call/r1/transcript.md)、
[case_63](../traces_glm/case_63/single_call/r1/transcript.md)、
[case_64](../traces_glm/case_64/single_call/r1/transcript.md)、
[case_65](../traces_glm/case_65/single_call/r1/transcript.md)。
