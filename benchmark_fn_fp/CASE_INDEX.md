# Benchmark 案例总索引

所有题库和 traces 共用 `case_数字` 编号。同一编号始终指向同一道题；来源记录在 case_map.json，部分题库共用统一公开目录。
本页是说明与答案侧索引，不作为模型输入。编号描述实验来源，不保证某种方法一定获胜。

当前 90 题；另有 24 题已归档。归档编号保留，不复用、不进入默认实验。

| 编号范围 | 数量 | 测什么 | 说明与题目 |
|---|---:|---|---|
| `case_01`–`case_35` | 34 | 原始 FN/FP benchmark | [说明](README.md) · [题目](triton_eval_cases/) |
| `case_36`–`case_61` | 26 | 无工具单次调用 vs 带工具验证 | [说明](single_call_vs_tools_challenges/README.md) · [题目](triton_eval_cases/) |
| `case_62`–`case_81` | 20 | Solo vs 多角色 debate | [说明](solo_vs_debate_challenges/README.md) · [题目](triton_eval_cases/) |
| `case_106`–`case_115` | 10 | 专门构造的真实 kernel 验证挑战（单独统计） | [说明](real_kernel_challenges/README.md) · [题目](triton_eval_cases/) |

已归档：`case_82`–`case_105`，共 24 题。[早期数值 pilot](archive/numerical_pilot/README.md) 保留题目、真值、旧 Opus 结果和 72 条 GLM 实验。

## 实验后的保留分组

这是后续整理状态，原始题库来源与全部实验统计保留。单次观察到增益不代表稳定优势；待归普通集合的案例目前仍保留在原来源组。

| 分组 | Case / traces |
|---|---|
| 成功案例：已观察到工具流程增益，单独保留 | [case_109](traces_glm/case_109/)、[case_114](traces_glm/case_114/) |
| 未体现增益，保留待归普通集合 | [case_106](traces_glm/case_106/)、[case_107](traces_glm/case_107/)、[case_108](traces_glm/case_108/)、[case_110](traces_glm/case_110/)、[case_111](traces_glm/case_111/)、[case_112](traces_glm/case_112/)、[case_113](traces_glm/case_113/)、[case_115](traces_glm/case_115/) |


已退休且不复用的编号：`case_03`。下一可用编号从 116 开始。
`real_kernel_challenges/` 保存真实源码衍生的专门构造挑战；标签针对固定实现与合同，结果单独统计。

每道题的完整对应关系如下。旧编号仅用于查阅历史记录，今后运行请使用新编号。

| 统一编号 | 状态 | 旧编号 | 题库 / 来源机制 | 公开题目 | GLM traces |
|---|---|---|---|---|---|
| `case_01` | 当前 | `case_01` | benchmark_fn_fp / 拒绝采样的负概率截断 | [kernel](triton_eval_cases/case_01/kernel.py) · [合同](triton_eval_cases/case_01/problem.txt) | [记录](traces_glm/case_01/) |
| `case_02` | 当前 | `case_02` | benchmark_fn_fp / Mamba 分块粒度与舍入 | [kernel](triton_eval_cases/case_02/kernel.py) · [合同](triton_eval_cases/case_02/problem.txt) | [记录](traces_glm/case_02/) |
| `case_04` | 当前 | `case_04` | benchmark_fn_fp / RMSNorm 近零相对误差 | [kernel](triton_eval_cases/case_04/kernel.py) · [合同](triton_eval_cases/case_04/problem.txt) | [记录](traces_glm/case_04/) |
| `case_05` | 当前 | `case_05` | benchmark_fn_fp / Top-p 并列候选的容差 | [kernel](triton_eval_cases/case_05/kernel.py) · [合同](triton_eval_cases/case_05/problem.txt) | [记录](traces_glm/case_05/) |
| `case_06` | 当前 | `case_06` | benchmark_fn_fp / Mamba 低精度状态误差累积 | [kernel](triton_eval_cases/case_06/kernel.py) · [合同](triton_eval_cases/case_06/problem.txt) | [记录](traces_glm/case_06/) |
| `case_07` | 当前 | `case_07` | benchmark_fn_fp / INT4 量化的合法误差 | [kernel](triton_eval_cases/case_07/kernel.py) · [合同](triton_eval_cases/case_07/problem.txt) | [记录](traces_glm/case_07/) |
| `case_08` | 当前 | `case_08` | benchmark_fn_fp / 随机舍入的统计正确性 | [kernel](triton_eval_cases/case_08/kernel.py) · [合同](triton_eval_cases/case_08/problem.txt) | [记录](traces_glm/case_08/) |
| `case_09` | 当前 | `case_09` | benchmark_fn_fp / Top-1 并列选择与下游误差稀释 | [kernel](triton_eval_cases/case_09/kernel.py) · [合同](triton_eval_cases/case_09/problem.txt) | [记录](traces_glm/case_09/) |
| `case_10` | 当前 | `case_10` | benchmark_fn_fp / Top-k 并列选择与下游误差稀释 | [kernel](triton_eval_cases/case_10/kernel.py) · [合同](triton_eval_cases/case_10/problem.txt) | [记录](traces_glm/case_10/) |
| `case_11` | 当前 | `case_11` | benchmark_fn_fp / RoPE 两种合法布局 | [kernel](triton_eval_cases/case_11/kernel.py) · [合同](triton_eval_cases/case_11/problem.txt) | [记录](traces_glm/case_11/) |
| `case_12` | 当前 | `case_12` | benchmark_fn_fp / RMSNorm epsilon 位置 | [kernel](triton_eval_cases/case_12/kernel.py) · [合同](triton_eval_cases/case_12/problem.txt) | [记录](traces_glm/case_12/) |
| `case_13` | 当前 | `case_13` | benchmark_fn_fp / GQA 头映射 | [kernel](triton_eval_cases/case_13/kernel.py) · [合同](triton_eval_cases/case_13/problem.txt) | [记录](traces_glm/case_13/) |
| `case_14` | 当前 | `case_14` | benchmark_fn_fp / INT8 量化的过期 scale | [kernel](triton_eval_cases/case_14/kernel.py) · [合同](triton_eval_cases/case_14/problem.txt) | [记录](traces_glm/case_14/) |
| `case_15` | 当前 | `case_15` | benchmark_fn_fp / INT8 量化最后一个短分组 | [kernel](triton_eval_cases/case_15/kernel.py) · [合同](triton_eval_cases/case_15/problem.txt) | [记录](traces_glm/case_15/) |
| `case_16` | 当前 | `case_16` | benchmark_fn_fp / 矩阵乘法 K 维尾块掩码 | [kernel](triton_eval_cases/case_16/kernel.py) · [合同](triton_eval_cases/case_16/problem.txt) | [记录](traces_glm/case_16/) |
| `case_17` | 当前 | `case_17` | benchmark_fn_fp / 分块 scan 的最后一个短块 | [kernel](triton_eval_cases/case_17/kernel.py) · [合同](triton_eval_cases/case_17/problem.txt) | [记录](traces_glm/case_17/) |
| `case_18` | 当前 | `case_18` | benchmark_fn_fp / Softmax 尾块归一化 | [kernel](triton_eval_cases/case_18/kernel.py) · [合同](triton_eval_cases/case_18/problem.txt) | [记录](traces_glm/case_18/) |
| `case_19` | 当前 | `case_19` | benchmark_fn_fp / 交叉熵 argmax 并列规则 | [kernel](triton_eval_cases/case_19/kernel.py) · [合同](triton_eval_cases/case_19/problem.txt) | [记录](traces_glm/case_19/) |
| `case_20` | 当前 | `case_20` | benchmark_fn_fp / LayerNorm epsilon 位置 | [kernel](triton_eval_cases/case_20/kernel.py) · [合同](triton_eval_cases/case_20/problem.txt) | [记录](traces_glm/case_20/) |
| `case_21` | 当前 | `case_21` | benchmark_fn_fp / Paged KV 页表映射 | [kernel](triton_eval_cases/case_21/kernel.py) · [合同](triton_eval_cases/case_21/problem.txt) | [记录](traces_glm/case_21/) |
| `case_22` | 当前 | `case_22` | benchmark_fn_fp / Split-K 原子累加顺序 | [kernel](triton_eval_cases/case_22/kernel.py) · [合同](triton_eval_cases/case_22/problem.txt) | [记录](traces_glm/case_22/) |
| `case_23` | 当前 | `case_23` | benchmark_fn_fp / NCHW/NHWC 合法布局差异 | [kernel](triton_eval_cases/case_23/kernel.py) · [合同](triton_eval_cases/case_23/problem.txt) | [记录](traces_glm/case_23/) |
| `case_24` | 当前 | `case_24` | benchmark_fn_fp / 余弦相似度近零误差 | [kernel](triton_eval_cases/case_24/kernel.py) · [合同](triton_eval_cases/case_24/problem.txt) | [记录](traces_glm/case_24/) |
| `case_25` | 当前 | `case_25` | benchmark_fn_fp / INT8 再量化舍入与累积 | [kernel](triton_eval_cases/case_25/kernel.py) · [合同](triton_eval_cases/case_25/problem.txt) | [记录](traces_glm/case_25/) |
| `case_26` | 当前 | `case_26` | benchmark_fn_fp / Top-k 合同允许的并列选择 | [kernel](triton_eval_cases/case_26/kernel.py) · [合同](triton_eval_cases/case_26/problem.txt) | [记录](traces_glm/case_26/) |
| `case_27` | 当前 | `case_27` | benchmark_fn_fp / 全掩码 Softmax 行 | [kernel](triton_eval_cases/case_27/kernel.py) · [合同](triton_eval_cases/case_27/problem.txt) | [记录](traces_glm/case_27/) |
| `case_28` | 当前 | `case_28` | benchmark_fn_fp / INT8 分位数量化的离群值 | [kernel](triton_eval_cases/case_28/kernel.py) · [合同](triton_eval_cases/case_28/problem.txt) | [记录](traces_glm/case_28/) |
| `case_29` | 当前 | `case_29` | benchmark_fn_fp / FP8 格式允许的舍入 | [kernel](triton_eval_cases/case_29/kernel.py) · [合同](triton_eval_cases/case_29/problem.txt) | [记录](traces_glm/case_29/) |
| `case_30` | 当前 | `case_30` | benchmark_fn_fp / MoE 专家并列路由 | [kernel](triton_eval_cases/case_30/kernel.py) · [合同](triton_eval_cases/case_30/problem.txt) | [记录](traces_glm/case_30/) |
| `case_31` | 当前 | `case_31` | benchmark_fn_fp / 变长 Attention 的整块分支 | [kernel](triton_eval_cases/case_31/kernel.py) · [合同](triton_eval_cases/case_31/problem.txt) | [记录](traces_glm/case_31/) |
| `case_32` | 当前 | `case_32` | benchmark_fn_fp / 稳定排序的并列顺序 | [kernel](triton_eval_cases/case_32/kernel.py) · [合同](triton_eval_cases/case_32/problem.txt) | [记录](traces_glm/case_32/) |
| `case_33` | 当前 | `case_33` | benchmark_fn_fp / GPTQ 最后一个量化分组 | [kernel](triton_eval_cases/case_33/kernel.py) · [合同](triton_eval_cases/case_33/problem.txt) | [记录](traces_glm/case_33/) |
| `case_34` | 当前 | `case_34` | benchmark_fn_fp / FP32 原子累加的逐位复现 | [kernel](triton_eval_cases/case_34/kernel.py) · [合同](triton_eval_cases/case_34/problem.txt) | [记录](traces_glm/case_34/) |
| `case_35` | 当前 | `case_35` | benchmark_fn_fp / 整数网格上原子累加的精确性 | [kernel](triton_eval_cases/case_35/kernel.py) · [合同](triton_eval_cases/case_35/problem.txt) | [记录](traces_glm/case_35/) |
| `case_36` | 当前 | `case_a` | single_call_vs_tools_challenges / 两路量化误差相加与抵消 | [kernel](triton_eval_cases/case_36/kernel.py) · [合同](triton_eval_cases/case_36/problem.txt) | [记录](traces_glm/case_36/) |
| `case_37` | 当前 | `case_b` | single_call_vs_tools_challenges / 两路量化误差相加与抵消 | [kernel](triton_eval_cases/case_37/kernel.py) · [合同](triton_eval_cases/case_37/problem.txt) | [记录](traces_glm/case_37/) |
| `case_38` | 当前 | `case_c` | single_call_vs_tools_challenges / Attention logits 量化误差 | [kernel](triton_eval_cases/case_38/kernel.py) · [合同](triton_eval_cases/case_38/problem.txt) | [记录](traces_glm/case_38/) |
| `case_39` | 当前 | `case_d` | single_call_vs_tools_challenges / Attention logits 量化误差 | [kernel](triton_eval_cases/case_39/kernel.py) · [合同](triton_eval_cases/case_39/problem.txt) | [记录](traces_glm/case_39/) |
| `case_40` | 当前 | `case_e` | single_call_vs_tools_challenges / FP16 递推舍入误差传播 | [kernel](triton_eval_cases/case_40/kernel.py) · [合同](triton_eval_cases/case_40/problem.txt) | [记录](traces_glm/case_40/) |
| `case_41` | 当前 | `case_f` | single_call_vs_tools_challenges / FP16 递推舍入误差传播 | [kernel](triton_eval_cases/case_41/kernel.py) · [合同](triton_eval_cases/case_41/problem.txt) | [记录](traces_glm/case_41/) |
| `case_42` | 当前 | `case_g` | single_call_vs_tools_challenges / FP32 求和顺序 | [kernel](triton_eval_cases/case_42/kernel.py) · [合同](triton_eval_cases/case_42/problem.txt) | [记录](traces_glm/case_42/) |
| `case_43` | 当前 | `case_h` | single_call_vs_tools_challenges / FP32 求和顺序 | [kernel](triton_eval_cases/case_43/kernel.py) · [合同](triton_eval_cases/case_43/problem.txt) | [记录](traces_glm/case_43/) |
| `case_44` | 当前 | `case_i` | single_call_vs_tools_challenges / LayerNorm 方差相消 | [kernel](triton_eval_cases/case_44/kernel.py) · [合同](triton_eval_cases/case_44/problem.txt) | [记录](traces_glm/case_44/) |
| `case_45` | 当前 | `case_j` | single_call_vs_tools_challenges / LayerNorm 方差相消 | [kernel](triton_eval_cases/case_45/kernel.py) · [合同](triton_eval_cases/case_45/problem.txt) | [记录](traces_glm/case_45/) |
| `case_46` | 当前 | `case_k` | single_call_vs_tools_challenges / 固定步数线性求解 | [kernel](triton_eval_cases/case_46/kernel.py) · [合同](triton_eval_cases/case_46/problem.txt) | [记录](traces_glm/case_46/) |
| `case_47` | 当前 | `case_l` | single_call_vs_tools_challenges / 固定步数线性求解 | [kernel](triton_eval_cases/case_47/kernel.py) · [合同](triton_eval_cases/case_47/problem.txt) | [记录](traces_glm/case_47/) |
| `case_48` | 当前 | `case_m` | single_call_vs_tools_challenges / Horner 多项式舍入 | [kernel](triton_eval_cases/case_48/kernel.py) · [合同](triton_eval_cases/case_48/problem.txt) | [记录](traces_glm/case_48/) |
| `case_49` | 当前 | `case_n` | single_call_vs_tools_challenges / Horner 多项式舍入 | [kernel](triton_eval_cases/case_49/kernel.py) · [合同](triton_eval_cases/case_49/problem.txt) | [记录](traces_glm/case_49/) |
| `case_50` | 当前 | `case_o` | single_call_vs_tools_challenges / 近共线投影与归一化 | [kernel](triton_eval_cases/case_50/kernel.py) · [合同](triton_eval_cases/case_50/problem.txt) | [记录](traces_glm/case_50/) |
| `case_51` | 当前 | `case_p` | single_call_vs_tools_challenges / 近共线投影与归一化 | [kernel](triton_eval_cases/case_51/kernel.py) · [合同](triton_eval_cases/case_51/problem.txt) | [记录](traces_glm/case_51/) |
| `case_52` | 当前 | `case_q` | single_call_vs_tools_challenges / 量化最近邻与路由 | [kernel](triton_eval_cases/case_52/kernel.py) · [合同](triton_eval_cases/case_52/problem.txt) | [记录](traces_glm/case_52/) |
| `case_53` | 当前 | `case_r` | single_call_vs_tools_challenges / 量化最近邻与路由 | [kernel](triton_eval_cases/case_53/kernel.py) · [合同](triton_eval_cases/case_53/problem.txt) | [记录](traces_glm/case_53/) |
| `case_54` | 当前 | `case_s` | single_call_vs_tools_challenges / 固定网格积分欠采样 | [kernel](triton_eval_cases/case_54/kernel.py) · [合同](triton_eval_cases/case_54/problem.txt) | [记录](traces_glm/case_54/) |
| `case_55` | 当前 | `case_t` | single_call_vs_tools_challenges / 固定网格积分欠采样 | [kernel](triton_eval_cases/case_55/kernel.py) · [合同](triton_eval_cases/case_55/problem.txt) | [记录](traces_glm/case_55/) |
| `case_56` | 当前 | `case_u` | single_call_vs_tools_challenges / Fourier 频带截断 | [kernel](triton_eval_cases/case_56/kernel.py) · [合同](triton_eval_cases/case_56/problem.txt) | [记录](traces_glm/case_56/) |
| `case_57` | 当前 | `case_v` | single_call_vs_tools_challenges / Fourier 频带截断 | [kernel](triton_eval_cases/case_57/kernel.py) · [合同](triton_eval_cases/case_57/problem.txt) | [记录](traces_glm/case_57/) |
| `case_58` | 当前 | `case_w` | single_call_vs_tools_challenges / 病态矩阵 logdet | [kernel](triton_eval_cases/case_58/kernel.py) · [合同](triton_eval_cases/case_58/problem.txt) | [记录](traces_glm/case_58/) |
| `case_59` | 当前 | `case_x` | single_call_vs_tools_challenges / 病态矩阵 logdet | [kernel](triton_eval_cases/case_59/kernel.py) · [合同](triton_eval_cases/case_59/problem.txt) | [记录](traces_glm/case_59/) |
| `case_60` | 当前 | `case_y` | single_call_vs_tools_challenges / RBF 距离展开相消 | [kernel](triton_eval_cases/case_60/kernel.py) · [合同](triton_eval_cases/case_60/problem.txt) | [记录](traces_glm/case_60/) |
| `case_61` | 当前 | `case_z` | single_call_vs_tools_challenges / RBF 距离展开相消 | [kernel](triton_eval_cases/case_61/kernel.py) · [合同](triton_eval_cases/case_61/problem.txt) | [记录](traces_glm/case_61/) |
| `case_62` | 当前 | `case_e01` | solo_vs_debate_challenges / 协方差回归的参考独立性 | [kernel](triton_eval_cases/case_62/kernel.py) · [合同](triton_eval_cases/case_62/problem.txt) | [记录](traces_glm/case_62/) |
| `case_63` | 当前 | `case_e02` | solo_vs_debate_challenges / 协方差回归的参考独立性 | [kernel](triton_eval_cases/case_63/kernel.py) · [合同](triton_eval_cases/case_63/problem.txt) | [记录](traces_glm/case_63/) |
| `case_64` | 当前 | `case_e03` | solo_vs_debate_challenges / 缓存状态序列覆盖 | [kernel](triton_eval_cases/case_64/kernel.py) · [合同](triton_eval_cases/case_64/problem.txt) | [记录](traces_glm/case_64/) |
| `case_65` | 当前 | `case_e04` | solo_vs_debate_challenges / 缓存状态序列覆盖 | [kernel](triton_eval_cases/case_65/kernel.py) · [合同](triton_eval_cases/case_65/problem.txt) | [记录](traces_glm/case_65/) |
| `case_66` | 当前 | `case_e05` | solo_vs_debate_challenges / 四阶联合分布 | [kernel](triton_eval_cases/case_66/kernel.py) · [合同](triton_eval_cases/case_66/problem.txt) | [记录](traces_glm/case_66/) |
| `case_67` | 当前 | `case_e06` | solo_vs_debate_challenges / 四阶联合分布 | [kernel](triton_eval_cases/case_67/kernel.py) · [合同](triton_eval_cases/case_67/problem.txt) | [记录](traces_glm/case_67/) |
| `case_68` | 当前 | `case_e07` | solo_vs_debate_challenges / 局部匹配与全局共同变换 | [kernel](triton_eval_cases/case_68/kernel.py) · [合同](triton_eval_cases/case_68/problem.txt) | [记录](traces_glm/case_68/) |
| `case_69` | 当前 | `case_e08` | solo_vs_debate_challenges / 局部匹配与全局共同变换 | [kernel](triton_eval_cases/case_69/kernel.py) · [合同](triton_eval_cases/case_69/problem.txt) | [记录](traces_glm/case_69/) |
| `case_70` | 当前 | `case_e09` | solo_vs_debate_challenges / 历史输出被后续更新覆盖 | [kernel](triton_eval_cases/case_70/kernel.py) · [合同](triton_eval_cases/case_70/problem.txt) | [记录](traces_glm/case_70/) |
| `case_71` | 当前 | `case_e10` | solo_vs_debate_challenges / 历史输出被后续更新覆盖 | [kernel](triton_eval_cases/case_71/kernel.py) · [合同](triton_eval_cases/case_71/problem.txt) | [记录](traces_glm/case_71/) |
| `case_72` | 当前 | `case_e11` | solo_vs_debate_challenges / 连续输入域的最坏情况 | [kernel](triton_eval_cases/case_72/kernel.py) · [合同](triton_eval_cases/case_72/problem.txt) | [记录](traces_glm/case_72/) |
| `case_73` | 当前 | `case_e12` | solo_vs_debate_challenges / 连续输入域的最坏情况 | [kernel](triton_eval_cases/case_73/kernel.py) · [合同](triton_eval_cases/case_73/problem.txt) | [记录](traces_glm/case_73/) |
| `case_74` | 当前 | `case_e13` | solo_vs_debate_challenges / FP64 求和参考失真 | [kernel](triton_eval_cases/case_74/kernel.py) · [合同](triton_eval_cases/case_74/problem.txt) | [记录](traces_glm/case_74/) |
| `case_75` | 当前 | `case_e14` | solo_vs_debate_challenges / FP64 求和参考失真 | [kernel](triton_eval_cases/case_75/kernel.py) · [合同](triton_eval_cases/case_75/problem.txt) | [记录](traces_glm/case_75/) |
| `case_76` | 当前 | `case_e15` | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](triton_eval_cases/case_76/kernel.py) · [合同](triton_eval_cases/case_76/problem.txt) | [记录](traces_glm/case_76/) |
| `case_77` | 当前 | `case_e16` | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](triton_eval_cases/case_77/kernel.py) · [合同](triton_eval_cases/case_77/problem.txt) | [记录](traces_glm/case_77/) |
| `case_78` | 当前 | `case_e17` | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](triton_eval_cases/case_78/kernel.py) · [合同](triton_eval_cases/case_78/problem.txt) | [记录](traces_glm/case_78/) |
| `case_79` | 当前 | `case_e18` | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](triton_eval_cases/case_79/kernel.py) · [合同](triton_eval_cases/case_79/problem.txt) | [记录](traces_glm/case_79/) |
| `case_80` | 当前 | `case_e19` | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](triton_eval_cases/case_80/kernel.py) · [合同](triton_eval_cases/case_80/problem.txt) | [记录](traces_glm/case_80/) |
| `case_81` | 当前 | `case_e20` | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](triton_eval_cases/case_81/kernel.py) · [合同](triton_eval_cases/case_81/problem.txt) | [记录](traces_glm/case_81/) |
| `case_82` | 已归档 | `case_01` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_82/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_82/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_82/) |
| `case_83` | 已归档 | `case_02` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_83/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_83/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_83/) |
| `case_84` | 已归档 | `case_03` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_84/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_84/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_84/) |
| `case_85` | 已归档 | `case_04` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_85/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_85/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_85/) |
| `case_86` | 已归档 | `case_05` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_86/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_86/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_86/) |
| `case_87` | 已归档 | `case_06` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_87/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_87/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_87/) |
| `case_88` | 已归档 | `case_07` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_88/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_88/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_88/) |
| `case_89` | 已归档 | `case_08` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_89/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_89/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_89/) |
| `case_90` | 已归档 | `case_09` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_90/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_90/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_90/) |
| `case_91` | 已归档 | `case_10` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_91/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_91/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_91/) |
| `case_92` | 已归档 | `case_11` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_92/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_92/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_92/) |
| `case_93` | 已归档 | `case_12` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_93/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_93/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_93/) |
| `case_94` | 已归档 | `case_13` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_94/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_94/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_94/) |
| `case_95` | 已归档 | `case_14` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_95/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_95/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_95/) |
| `case_96` | 已归档 | `case_15` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_96/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_96/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_96/) |
| `case_97` | 已归档 | `case_16` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_97/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_97/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_97/) |
| `case_98` | 已归档 | `case_17` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_98/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_98/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_98/) |
| `case_99` | 已归档 | `case_18` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_99/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_99/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_99/) |
| `case_100` | 已归档 | `case_19` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_100/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_100/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_100/) |
| `case_101` | 已归档 | `case_20` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_101/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_101/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_101/) |
| `case_102` | 已归档 | `case_21` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_102/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_102/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_102/) |
| `case_103` | 已归档 | `case_22` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_103/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_103/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_103/) |
| `case_104` | 已归档 | `case_23` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_104/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_104/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_104/) |
| `case_105` | 已归档 | `case_24` | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_105/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_105/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_105/) |
| `case_106` | 当前 · 未体现增益，保留待归普通集合 | `case_106` | real_kernel_challenges / 状态滑动窗口并行读写与别名 | [kernel](triton_eval_cases/case_106/kernel.py) · [合同](triton_eval_cases/case_106/problem.txt) | [记录](traces_glm/case_106/) |
| `case_107` | 当前 · 未体现增益，保留待归普通集合 | `case_107` | real_kernel_challenges / 状态写回：张量状态支持遗漏（补审发现） | [kernel](triton_eval_cases/case_107/kernel.py) · [合同](triton_eval_cases/case_107/problem.txt) | [记录](traces_glm/case_107/) |
| `case_108` | 当前 · 未体现增益，保留待归普通集合 | `case_108` | real_kernel_challenges / RMSNorm：逻辑视图支持遗漏（补审发现） | [kernel](triton_eval_cases/case_108/kernel.py) · [合同](triton_eval_cases/case_108/problem.txt) | [记录](traces_glm/case_108/) |
| `case_109` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | `case_109` | real_kernel_challenges / RMSNorm：半精度梯度归约误差 | [kernel](triton_eval_cases/case_109/kernel.py) · [合同](triton_eval_cases/case_109/problem.txt) | [记录](traces_glm/case_109/) |
| `case_110` | 当前 · 未体现增益，保留待归普通集合 | `case_110` | real_kernel_challenges / 状态更新：并行读写与张量状态均已修复 | [kernel](triton_eval_cases/case_110/kernel.py) · [合同](triton_eval_cases/case_110/problem.txt) | [记录](traces_glm/case_110/) |
| `case_111` | 当前 · 未体现增益，保留待归普通集合 | `case_111` | real_kernel_challenges / RMSNorm backward：归约与逻辑视图均已修复 | [kernel](triton_eval_cases/case_111/kernel.py) · [合同](triton_eval_cases/case_111/problem.txt) | [记录](traces_glm/case_111/) |
| `case_112` | 当前 · 未体现增益，保留待归普通集合 | `case_112` | real_kernel_challenges / 分段递推：跨 chunk 状态传递 | [kernel](triton_eval_cases/case_112/kernel.py) · [合同](triton_eval_cases/case_112/problem.txt) | [记录](traces_glm/case_112/) |
| `case_113` | 当前 · 未体现增益，保留待归普通集合 | `case_113` | real_kernel_challenges / 分段递推：段标识选取错误 | [kernel](triton_eval_cases/case_113/kernel.py) · [合同](triton_eval_cases/case_113/problem.txt) | [记录](traces_glm/case_113/) |
| `case_114` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | `case_114` | real_kernel_challenges / 分页 Attention：窗口外 V 遮罩遗漏 | [kernel](triton_eval_cases/case_114/kernel.py) · [合同](triton_eval_cases/case_114/problem.txt) | [记录](traces_glm/case_114/) |
| `case_115` | 当前 · 未体现增益，保留待归普通集合 | `case_115` | real_kernel_challenges / 分页 Attention：窗口外 V 遮罩修复 | [kernel](triton_eval_cases/case_115/kernel.py) · [合同](triton_eval_cases/case_115/problem.txt) | [记录](traces_glm/case_115/) |

## 如何读 traces

`traces_glm/<case>/<arm>/<trial>/` 中，`single_call` 是无工具单次调用，`solo` 是单 agent 加工具，`debate` 是多角色加工具。各组批次目录统一为 `r1`、`r2` 等运行序号；原批次名保留在 `trace_meta.json` 的 `original_trial` 中。同名 `r1` 不代表模型、预算或实验配置相同。
[GLM 逐次运行索引](traces_glm/INDEX.md) 保留不同配置、复测与失败尝试；题库编号相同不代表三组都已跑齐。
目录已统一编号；历史请求、回复和 probe 代码保留原始字节；目录和 metadata 的身份、迁移字段按索引更新。原始内容中的旧编号不是另一道题，读取程序根据当前目录与本注册表核对身份。
case_36–case_37 已并入单次调用与工具比较组；其 Opus 和 GLM 原始记录仍在 traces_opus5 与 traces_glm，历史来源字段不改写。
早期数值 pilot（case_82–case_105）和它的 72 条 GLM 记录移入 archive/numerical_pilot，已从当前 GLM 索引与默认评分移出；过去 104 题的历史报告保留原结论。

## 文件清理说明

两组 challenges 各保留一份 README 和 `private_data/`。后者保存答案、搜索过程和 GPU 真值校验；这些是实验可复现性依据。
当前题目统一在 `triton_eval_cases/`，构造、校验与报告程序统一在 `eval_scripts/` 的对应目录；pilot 的公开题目留在归档中。
零散实验说明已合并进各组 README；旧逻辑数据集名只用于兼容历史 traces 和旧命令。可从原始 traces 重建的 scoreboard JSON 已清理；报告脚本默认更新 README 中的结果表，如需机器可读汇总可显式请求 JSON。原始模型调用与 GPU 记录不属于可重建汇总。

本页由 `eval_scripts/index_cases.py` 根据 `case_map.json` 生成。
