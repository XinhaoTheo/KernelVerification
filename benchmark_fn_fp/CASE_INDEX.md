# Benchmark 案例总索引

本页、题目目录和 traces 使用同一套当前 `case_数字` 编号。来源和历史映射保存在 case_map.json。
本页是说明与答案侧索引，不作为模型输入。编号描述实验来源，不保证某种方法一定获胜。

当前 102 题；另有 72 题已归档。当前编号：`case_01`–`case_102`。归档题不进入默认实验，也不计入当前题数。

| 编号范围 | 数量 | 测什么 | 说明与题目 |
|---|---:|---|---|
| `case_01`–`case_32` | 32 | 原始 FN/FP benchmark | [说明](README.md) · [题目](triton_eval_cases/) |
| `case_33`–`case_102` | 70 | 专门构造的真实 kernel 验证挑战（单独统计） | [说明](real_kernel_challenges/README.md) · [题目](triton_eval_cases/) |

| 归档 | 编号范围 | 数量 | 保留内容 |
|---|---|---:|---|
| [按指定输入设计的旧案例](archive/input_specific/README.md) | `case_103`–`case_150` | 48 | 题目、原始标签、构造记录与历史 traces；不计入当前题库 |
| [早期数值 pilot](archive/numerical_pilot/README.md) | `case_151`–`case_174` | 24 | 题目、原始标签、构造记录与历史 traces；不计入当前题库 |

## 实验后的保留分组

这是后续整理状态，原始题库来源与全部实验统计保留。单次观察到增益不代表稳定优势；待归普通集合的案例目前仍保留在原来源组。

| 分组 | Case / traces |
|---|---|
| 成功案例：已观察到工具流程增益，单独保留 | [case_36](traces_glm/case_36/)、[case_41](traces_glm/case_41/)、[case_54](traces_glm/case_54/)、[case_55](traces_glm/case_55/)、[case_62](traces_glm/case_62/)、[case_68](traces_glm/case_68/)、[case_74](traces_glm/case_74/)、[case_76](traces_glm/case_76/)、[case_78](traces_glm/case_78/)、[case_81](traces_glm/case_81/)、[case_86](traces_glm/case_86/)、[case_90](traces_glm/case_90/)、[case_94](traces_glm/case_94/)、[case_100](traces_glm/case_100/) |
| 未体现增益，保留待归普通集合 | [case_33](traces_glm/case_33/)、[case_34](traces_glm/case_34/)、[case_35](traces_glm/case_35/)、[case_37](traces_glm/case_37/)、[case_38](traces_glm/case_38/)、[case_39](traces_glm/case_39/)、[case_40](traces_glm/case_40/)、[case_42](traces_glm/case_42/)、[case_43](traces_glm/case_43/)、[case_44](traces_glm/case_44/)、[case_45](traces_glm/case_45/)、[case_46](traces_glm/case_46/)、[case_47](traces_glm/case_47/)、[case_48](traces_glm/case_48/)、[case_49](traces_glm/case_49/)、[case_50](traces_glm/case_50/)、[case_51](traces_glm/case_51/)、[case_52](traces_glm/case_52/)、[case_53](traces_glm/case_53/)、[case_56](traces_glm/case_56/)、[case_57](traces_glm/case_57/)、[case_58](traces_glm/case_58/)、[case_59](traces_glm/case_59/)、[case_60](traces_glm/case_60/)、[case_61](traces_glm/case_61/)、[case_63](traces_glm/case_63/)、[case_64](traces_glm/case_64/)、[case_65](traces_glm/case_65/)、[case_66](traces_glm/case_66/)、[case_67](traces_glm/case_67/)、[case_69](traces_glm/case_69/)、[case_70](traces_glm/case_70/)、[case_71](traces_glm/case_71/)、[case_72](traces_glm/case_72/)、[case_73](traces_glm/case_73/)、[case_75](traces_glm/case_75/)、[case_77](traces_glm/case_77/)、[case_79](traces_glm/case_79/)、[case_80](traces_glm/case_80/)、[case_82](traces_glm/case_82/)、[case_83](traces_glm/case_83/)、[case_84](traces_glm/case_84/)、[case_85](traces_glm/case_85/)、[case_87](traces_glm/case_87/)、[case_88](traces_glm/case_88/)、[case_89](traces_glm/case_89/)、[case_91](traces_glm/case_91/)、[case_92](traces_glm/case_92/)、[case_93](traces_glm/case_93/)、[case_95](traces_glm/case_95/)、[case_96](traces_glm/case_96/)、[case_97](traces_glm/case_97/)、[case_98](traces_glm/case_98/)、[case_99](traces_glm/case_99/)、[case_101](traces_glm/case_101/)、[case_102](traces_glm/case_102/) |


新增当前题接在当前题库末尾编号；下一当前编号为 `case_103`。归档编号不表示当前题数。
`real_kernel_challenges/` 保存真实源码衍生的专门构造挑战；标签针对固定实现与合同，结果单独统计。

下表只使用当前编号。归档条目单独标明，不属于当前实验题库。

| 编号 | 状态 | 题库 / 来源机制 | 公开题目 | GLM traces |
|---|---|---|---|---|
| `case_01` | 当前 | benchmark_fn_fp / 拒绝采样的负概率截断 | [kernel](triton_eval_cases/case_01/kernel.py) · [合同](triton_eval_cases/case_01/problem.txt) | [记录](traces_glm/case_01/) |
| `case_02` | 当前 | benchmark_fn_fp / Mamba 分块粒度与舍入 | [kernel](triton_eval_cases/case_02/kernel.py) · [合同](triton_eval_cases/case_02/problem.txt) | [记录](traces_glm/case_02/) |
| `case_03` | 当前 | benchmark_fn_fp / RMSNorm 近零相对误差 | [kernel](triton_eval_cases/case_03/kernel.py) · [合同](triton_eval_cases/case_03/problem.txt) | [记录](traces_glm/case_03/) |
| `case_04` | 当前 | benchmark_fn_fp / Top-p 并列候选的容差 | [kernel](triton_eval_cases/case_04/kernel.py) · [合同](triton_eval_cases/case_04/problem.txt) | [记录](traces_glm/case_04/) |
| `case_05` | 当前 | benchmark_fn_fp / Mamba 低精度状态误差累积 | [kernel](triton_eval_cases/case_05/kernel.py) · [合同](triton_eval_cases/case_05/problem.txt) | [记录](traces_glm/case_05/) |
| `case_06` | 当前 | benchmark_fn_fp / INT4 量化的合法误差 | [kernel](triton_eval_cases/case_06/kernel.py) · [合同](triton_eval_cases/case_06/problem.txt) | [记录](traces_glm/case_06/) |
| `case_07` | 当前 | benchmark_fn_fp / 随机舍入的统计正确性 | [kernel](triton_eval_cases/case_07/kernel.py) · [合同](triton_eval_cases/case_07/problem.txt) | [记录](traces_glm/case_07/) |
| `case_08` | 当前 | benchmark_fn_fp / Top-1 并列选择与下游误差稀释 | [kernel](triton_eval_cases/case_08/kernel.py) · [合同](triton_eval_cases/case_08/problem.txt) | [记录](traces_glm/case_08/) |
| `case_09` | 当前 | benchmark_fn_fp / Top-k 并列选择与下游误差稀释 | [kernel](triton_eval_cases/case_09/kernel.py) · [合同](triton_eval_cases/case_09/problem.txt) | [记录](traces_glm/case_09/) |
| `case_10` | 当前 | benchmark_fn_fp / RoPE 两种合法布局 | [kernel](triton_eval_cases/case_10/kernel.py) · [合同](triton_eval_cases/case_10/problem.txt) | [记录](traces_glm/case_10/) |
| `case_11` | 当前 | benchmark_fn_fp / RMSNorm epsilon 位置 | [kernel](triton_eval_cases/case_11/kernel.py) · [合同](triton_eval_cases/case_11/problem.txt) | [记录](traces_glm/case_11/) |
| `case_12` | 当前 | benchmark_fn_fp / GQA 头映射 | [kernel](triton_eval_cases/case_12/kernel.py) · [合同](triton_eval_cases/case_12/problem.txt) | [记录](traces_glm/case_12/) |
| `case_13` | 当前 | benchmark_fn_fp / INT8 量化的过期 scale | [kernel](triton_eval_cases/case_13/kernel.py) · [合同](triton_eval_cases/case_13/problem.txt) | [记录](traces_glm/case_13/) |
| `case_14` | 当前 | benchmark_fn_fp / INT8 量化最后一个短分组 | [kernel](triton_eval_cases/case_14/kernel.py) · [合同](triton_eval_cases/case_14/problem.txt) | [记录](traces_glm/case_14/) |
| `case_15` | 当前 | benchmark_fn_fp / 矩阵乘法 K 维尾块掩码 | [kernel](triton_eval_cases/case_15/kernel.py) · [合同](triton_eval_cases/case_15/problem.txt) | [记录](traces_glm/case_15/) |
| `case_16` | 当前 | benchmark_fn_fp / 分块 scan 的最后一个短块 | [kernel](triton_eval_cases/case_16/kernel.py) · [合同](triton_eval_cases/case_16/problem.txt) | [记录](traces_glm/case_16/) |
| `case_17` | 当前 | benchmark_fn_fp / Softmax 尾块归一化 | [kernel](triton_eval_cases/case_17/kernel.py) · [合同](triton_eval_cases/case_17/problem.txt) | [记录](traces_glm/case_17/) |
| `case_18` | 当前 | benchmark_fn_fp / 交叉熵 argmax 并列规则 | [kernel](triton_eval_cases/case_18/kernel.py) · [合同](triton_eval_cases/case_18/problem.txt) | [记录](traces_glm/case_18/) |
| `case_19` | 当前 | benchmark_fn_fp / LayerNorm epsilon 位置 | [kernel](triton_eval_cases/case_19/kernel.py) · [合同](triton_eval_cases/case_19/problem.txt) | [记录](traces_glm/case_19/) |
| `case_20` | 当前 | benchmark_fn_fp / Paged KV 页表映射 | [kernel](triton_eval_cases/case_20/kernel.py) · [合同](triton_eval_cases/case_20/problem.txt) | [记录](traces_glm/case_20/) |
| `case_21` | 当前 | benchmark_fn_fp / Split-K 原子累加顺序 | [kernel](triton_eval_cases/case_21/kernel.py) · [合同](triton_eval_cases/case_21/problem.txt) | [记录](traces_glm/case_21/) |
| `case_22` | 当前 | benchmark_fn_fp / NCHW/NHWC 合法布局差异 | [kernel](triton_eval_cases/case_22/kernel.py) · [合同](triton_eval_cases/case_22/problem.txt) | [记录](traces_glm/case_22/) |
| `case_23` | 当前 | benchmark_fn_fp / 余弦相似度近零误差 | [kernel](triton_eval_cases/case_23/kernel.py) · [合同](triton_eval_cases/case_23/problem.txt) | [记录](traces_glm/case_23/) |
| `case_24` | 当前 | benchmark_fn_fp / INT8 再量化舍入与累积 | [kernel](triton_eval_cases/case_24/kernel.py) · [合同](triton_eval_cases/case_24/problem.txt) | [记录](traces_glm/case_24/) |
| `case_25` | 当前 | benchmark_fn_fp / Top-k 合同允许的并列选择 | [kernel](triton_eval_cases/case_25/kernel.py) · [合同](triton_eval_cases/case_25/problem.txt) | [记录](traces_glm/case_25/) |
| `case_26` | 当前 | benchmark_fn_fp / 全掩码 Softmax 行 | [kernel](triton_eval_cases/case_26/kernel.py) · [合同](triton_eval_cases/case_26/problem.txt) | [记录](traces_glm/case_26/) |
| `case_27` | 当前 | benchmark_fn_fp / INT8 分位数量化的离群值 | [kernel](triton_eval_cases/case_27/kernel.py) · [合同](triton_eval_cases/case_27/problem.txt) | [记录](traces_glm/case_27/) |
| `case_28` | 当前 | benchmark_fn_fp / FP8 格式允许的舍入 | [kernel](triton_eval_cases/case_28/kernel.py) · [合同](triton_eval_cases/case_28/problem.txt) | [记录](traces_glm/case_28/) |
| `case_29` | 当前 | benchmark_fn_fp / MoE 专家并列路由 | [kernel](triton_eval_cases/case_29/kernel.py) · [合同](triton_eval_cases/case_29/problem.txt) | [记录](traces_glm/case_29/) |
| `case_30` | 当前 | benchmark_fn_fp / 变长 Attention 的整块分支 | [kernel](triton_eval_cases/case_30/kernel.py) · [合同](triton_eval_cases/case_30/problem.txt) | [记录](traces_glm/case_30/) |
| `case_31` | 当前 | benchmark_fn_fp / 稳定排序的并列顺序 | [kernel](triton_eval_cases/case_31/kernel.py) · [合同](triton_eval_cases/case_31/problem.txt) | [记录](traces_glm/case_31/) |
| `case_32` | 当前 | benchmark_fn_fp / GPTQ 最后一个量化分组 | [kernel](triton_eval_cases/case_32/kernel.py) · [合同](triton_eval_cases/case_32/problem.txt) | [记录](traces_glm/case_32/) |
| `case_33` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 状态滑动窗口并行读写与别名 | [kernel](triton_eval_cases/case_33/kernel.py) · [合同](triton_eval_cases/case_33/problem.txt) | [记录](traces_glm/case_33/) |
| `case_34` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 状态写回：张量状态支持遗漏（补审发现） | [kernel](triton_eval_cases/case_34/kernel.py) · [合同](triton_eval_cases/case_34/problem.txt) | [记录](traces_glm/case_34/) |
| `case_35` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / RMSNorm：逻辑视图支持遗漏（补审发现） | [kernel](triton_eval_cases/case_35/kernel.py) · [合同](triton_eval_cases/case_35/problem.txt) | [记录](traces_glm/case_35/) |
| `case_36` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / RMSNorm：半精度梯度归约误差 | [kernel](triton_eval_cases/case_36/kernel.py) · [合同](triton_eval_cases/case_36/problem.txt) | [记录](traces_glm/case_36/) |
| `case_37` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 状态更新：并行读写与张量状态均已修复 | [kernel](triton_eval_cases/case_37/kernel.py) · [合同](triton_eval_cases/case_37/problem.txt) | [记录](traces_glm/case_37/) |
| `case_38` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / RMSNorm backward：归约与逻辑视图均已修复 | [kernel](triton_eval_cases/case_38/kernel.py) · [合同](triton_eval_cases/case_38/problem.txt) | [记录](traces_glm/case_38/) |
| `case_39` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 分段递推：跨 chunk 状态传递 | [kernel](triton_eval_cases/case_39/kernel.py) · [合同](triton_eval_cases/case_39/problem.txt) | [记录](traces_glm/case_39/) |
| `case_40` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 分段递推：段标识选取错误 | [kernel](triton_eval_cases/case_40/kernel.py) · [合同](triton_eval_cases/case_40/problem.txt) | [记录](traces_glm/case_40/) |
| `case_41` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / 分页 Attention：窗口外 V 遮罩遗漏 | [kernel](triton_eval_cases/case_41/kernel.py) · [合同](triton_eval_cases/case_41/problem.txt) | [记录](traces_glm/case_41/) |
| `case_42` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 分页 Attention：窗口外 V 遮罩修复 | [kernel](triton_eval_cases/case_42/kernel.py) · [合同](triton_eval_cases/case_42/problem.txt) | [记录](traces_glm/case_42/) |
| `case_43` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Split-K GEMM：半精度中间缓冲 | [kernel](triton_eval_cases/case_43/kernel.py) · [合同](triton_eval_cases/case_43/problem.txt) | [记录](traces_glm/case_43/) |
| `case_44` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Split-K GEMM：FP32 中间缓冲 | [kernel](triton_eval_cases/case_44/kernel.py) · [合同](triton_eval_cases/case_44/problem.txt) | [记录](traces_glm/case_44/) |
| `case_45` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / MoE grouped GEMM：完整路由与尾组调度 | [kernel](triton_eval_cases/case_45/kernel.py) · [合同](triton_eval_cases/case_45/problem.txt) | [记录](traces_glm/case_45/) |
| `case_46` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / MoE grouped GEMM：尾组输出 tile 遗漏 | [kernel](triton_eval_cases/case_46/kernel.py) · [合同](triton_eval_cases/case_46/problem.txt) | [记录](traces_glm/case_46/) |
| `case_47` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / INT4 GEMM：完整 GPTQ 到 Marlin 布局转换 | [kernel](triton_eval_cases/case_47/kernel.py) · [合同](triton_eval_cases/case_47/problem.txt) | [记录](traces_glm/case_47/) |
| `case_48` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / INT4 GEMM：重排器与消费端格式不匹配 | [kernel](triton_eval_cases/case_48/kernel.py) · [合同](triton_eval_cases/case_48/problem.txt) | [记录](traces_glm/case_48/) |
| `case_49` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 时间融合 Stencil：有限域镜像展开 | [kernel](triton_eval_cases/case_49/kernel.py) · [合同](triton_eval_cases/case_49/problem.txt) | [记录](traces_glm/case_49/) |
| `case_50` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 时间融合 Stencil：边界扩展与融合不兼容 | [kernel](triton_eval_cases/case_50/kernel.py) · [合同](triton_eval_cases/case_50/problem.txt) | [记录](traces_glm/case_50/) |
| `case_51` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 整数重定标：两级舍入与串联进位 | [kernel](triton_eval_cases/case_51/kernel.py) · [合同](triton_eval_cases/case_51/problem.txt) | [记录](traces_glm/case_51/) |
| `case_52` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 整数重定标：拆分进位的依赖错误 | [kernel](triton_eval_cases/case_52/kernel.py) · [合同](triton_eval_cases/case_52/problem.txt) | [记录](traces_glm/case_52/) |
| `case_53` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / erf 快速近似：全有限 FP16 域精度 | [kernel](triton_eval_cases/case_53/kernel.py) · [合同](triton_eval_cases/case_53/problem.txt) | [记录](traces_glm/case_53/) |
| `case_54` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / erf 快速近似：系数表序列化精度 | [kernel](triton_eval_cases/case_54/kernel.py) · [合同](triton_eval_cases/case_54/problem.txt) | [记录](traces_glm/case_54/) |
| `case_55` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / GELU：五位系数的全域精度 | [kernel](triton_eval_cases/case_55/kernel.py) · [合同](triton_eval_cases/case_55/problem.txt) | [记录](traces_glm/case_55/) |
| `case_56` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / GELU：归一化参数序列化误差 | [kernel](triton_eval_cases/case_56/kernel.py) · [合同](triton_eval_cases/case_56/problem.txt) | [记录](traces_glm/case_56/) |
| `case_57` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / SiLU：稳定尾项与完整换底常数 | [kernel](triton_eval_cases/case_57/kernel.py) · [合同](triton_eval_cases/case_57/problem.txt) | [记录](traces_glm/case_57/) |
| `case_58` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / SiLU：指数换底常数保存精度 | [kernel](triton_eval_cases/case_58/kernel.py) · [合同](triton_eval_cases/case_58/problem.txt) | [记录](traces_glm/case_58/) |
| `case_59` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 正弦：分拆常数与融合参数缩减 | [kernel](triton_eval_cases/case_59/kernel.py) · [合同](triton_eval_cases/case_59/problem.txt) | [记录](traces_glm/case_59/) |
| `case_60` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 正弦：分步舍入的稀疏超差 | [kernel](triton_eval_cases/case_60/kernel.py) · [合同](triton_eval_cases/case_60/problem.txt) | [记录](traces_glm/case_60/) |
| `case_61` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 倒平方根归一化：三次 Newton 修正 | [kernel](triton_eval_cases/case_61/kernel.py) · [合同](triton_eval_cases/case_61/problem.txt) | [记录](traces_glm/case_61/) |
| `case_62` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / 倒平方根归一化：两次 Newton 修正 | [kernel](triton_eval_cases/case_62/kernel.py) · [合同](triton_eval_cases/case_62/problem.txt) | [记录](traces_glm/case_62/) |
| `case_63` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Softplus：范围缩减与七阶对数近似 | [kernel](triton_eval_cases/case_63/kernel.py) · [合同](triton_eval_cases/case_63/problem.txt) | [记录](traces_glm/case_63/) |
| `case_64` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Softplus：范围缩减与五阶对数近似 | [kernel](triton_eval_cases/case_64/kernel.py) · [合同](triton_eval_cases/case_64/problem.txt) | [记录](traces_glm/case_64/) |
| `case_65` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Pseudo-Huber：消去误差补偿 | [kernel](triton_eval_cases/case_65/kernel.py) · [合同](triton_eval_cases/case_65/problem.txt) | [记录](traces_glm/case_65/) |
| `case_66` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Pseudo-Huber：直接相减 | [kernel](triton_eval_cases/case_66/kernel.py) · [合同](triton_eval_cases/case_66/problem.txt) | [记录](traces_glm/case_66/) |
| `case_67` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / FastDivmod：完整 uint32 域的向下倒数修正 | [kernel](triton_eval_cases/case_67/kernel.py) · [合同](triton_eval_cases/case_67/problem.txt) | [记录](traces_glm/case_67/) |
| `case_68` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / FastDivmod：窄域向上倒数用于完整 uint32 | [kernel](triton_eval_cases/case_68/kernel.py) · [合同](triton_eval_cases/case_68/problem.txt) | [记录](traces_glm/case_68/) |
| `case_69` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Walsh–Hadamard：正确 sequency 排序 | [kernel](triton_eval_cases/case_69/kernel.py) · [合同](triton_eval_cases/case_69/problem.txt) | [记录](traces_glm/case_69/) |
| `case_70` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Walsh–Hadamard：重排组合顺序错误 | [kernel](triton_eval_cases/case_70/kernel.py) · [合同](triton_eval_cases/case_70/problem.txt) | [记录](traces_glm/case_70/) |
| `case_71` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Argmax 位编码：数值相等时保留首个索引 | [kernel](triton_eval_cases/case_71/kernel.py) · [合同](triton_eval_cases/case_71/problem.txt) | [记录](traces_glm/case_71/) |
| `case_72` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Argmax 位编码：正负零的相等关系遗漏 | [kernel](triton_eval_cases/case_72/kernel.py) · [合同](triton_eval_cases/case_72/problem.txt) | [记录](traces_glm/case_72/) |
| `case_73` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Atan：五位有效数字的分段有理逼近 | [kernel](triton_eval_cases/case_73/kernel.py) · [合同](triton_eval_cases/case_73/problem.txt) | [记录](traces_glm/case_73/) |
| `case_74` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / Atan：四位有效数字的分段有理逼近 | [kernel](triton_eval_cases/case_74/kernel.py) · [合同](triton_eval_cases/case_74/problem.txt) | [记录](traces_glm/case_74/) |
| `case_75` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Sigmoid：FP64计算后正确舍入到FP16 | [kernel](triton_eval_cases/case_75/kernel.py) · [合同](triton_eval_cases/case_75/problem.txt) | [记录](traces_glm/case_75/) |
| `case_76` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / Sigmoid：FP32计算后的FP16舍入边界 | [kernel](triton_eval_cases/case_76/kernel.py) · [合同](triton_eval_cases/case_76/problem.txt) | [记录](traces_glm/case_76/) |
| `case_77` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / NTT：融合蝶形的Montgomery表示转换 | [kernel](triton_eval_cases/case_77/kernel.py) · [合同](triton_eval_cases/case_77/problem.txt) | [记录](traces_glm/case_77/) |
| `case_78` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / NTT：融合系数多保留一个Montgomery因子 | [kernel](triton_eval_cases/case_78/kernel.py) · [合同](triton_eval_cases/case_78/problem.txt) | [记录](traces_glm/case_78/) |
| `case_79` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / CRC：并行分块的反射多项式合并 | [kernel](triton_eval_cases/case_79/kernel.py) · [合同](triton_eval_cases/case_79/problem.txt) | [记录](traces_glm/case_79/) |
| `case_80` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / CRC：多项式幂的表示不一致 | [kernel](triton_eval_cases/case_80/kernel.py) · [合同](triton_eval_cases/case_80/problem.txt) | [记录](traces_glm/case_80/) |
| `case_81` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / 复数除法：FP32 直接乘积与除法 | [kernel](triton_eval_cases/case_81/kernel.py) · [合同](triton_eval_cases/case_81/problem.txt) | [记录](traces_glm/case_81/) |
| `case_82` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 复数除法：比值重写与分量抵消 | [kernel](triton_eval_cases/case_82/kernel.py) · [合同](triton_eval_cases/case_82/problem.txt) | [记录](traces_glm/case_82/) |
| `case_83` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / INT8 重量化：整数单次舍入 | [kernel](triton_eval_cases/case_83/kernel.py) · [合同](triton_eval_cases/case_83/problem.txt) | [记录](traces_glm/case_83/) |
| `case_84` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / INT8 重量化：FP64 中间乘积与精确舍入 | [kernel](triton_eval_cases/case_84/kernel.py) · [合同](triton_eval_cases/case_84/problem.txt) | [记录](traces_glm/case_84/) |
| `case_85` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 并行仿射递推：顺序一致的分块合并 | [kernel](triton_eval_cases/case_85/kernel.py) · [合同](triton_eval_cases/case_85/problem.txt) | [记录](traces_glm/case_85/) |
| `case_86` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / 并行仿射递推：分块组合顺序错误 | [kernel](triton_eval_cases/case_86/kernel.py) · [合同](triton_eval_cases/case_86/problem.txt) | [记录](traces_glm/case_86/) |
| `case_87` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Winograd 卷积：一致的变换排列 | [kernel](triton_eval_cases/case_87/kernel.py) · [合同](triton_eval_cases/case_87/problem.txt) | [记录](traces_glm/case_87/) |
| `case_88` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Winograd 卷积：变换排列不匹配 | [kernel](triton_eval_cases/case_88/kernel.py) · [合同](triton_eval_cases/case_88/problem.txt) | [记录](traces_glm/case_88/) |
| `case_89` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / 实数 FFT：频率排列与共轭配对 | [kernel](triton_eval_cases/case_89/kernel.py) · [合同](triton_eval_cases/case_89/problem.txt) | [记录](traces_glm/case_89/) |
| `case_90` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / 实数 FFT：频率排列与共轭配对错误 | [kernel](triton_eval_cases/case_90/kernel.py) · [合同](triton_eval_cases/case_90/problem.txt) | [记录](traces_glm/case_90/) |
| `case_91` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / BF16 exp2：区间缩减后的系数精度 | [kernel](triton_eval_cases/case_91/kernel.py) · [合同](triton_eval_cases/case_91/problem.txt) | [记录](traces_glm/case_91/) |
| `case_92` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / BF16 exp2：区间缩减后的系数精度 | [kernel](triton_eval_cases/case_92/kernel.py) · [合同](triton_eval_cases/case_92/problem.txt) | [记录](traces_glm/case_92/) |
| `case_93` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / FP16 log1p：区间缩减与补偿重构 | [kernel](triton_eval_cases/case_93/kernel.py) · [合同](triton_eval_cases/case_93/problem.txt) | [记录](traces_glm/case_93/) |
| `case_94` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / FP16 log1p：区间缩减与补偿重构 | [kernel](triton_eval_cases/case_94/kernel.py) · [合同](triton_eval_cases/case_94/problem.txt) | [记录](traces_glm/case_94/) |
| `case_95` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / AES 字节替换：复合有限域的基转换 | [kernel](triton_eval_cases/case_95/kernel.py) · [合同](triton_eval_cases/case_95/problem.txt) | [记录](traces_glm/case_95/) |
| `case_96` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / AES 字节替换：复合有限域的基转换 | [kernel](triton_eval_cases/case_96/kernel.py) · [合同](triton_eval_cases/case_96/problem.txt) | [记录](traces_glm/case_96/) |
| `case_97` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / expm1 range reduction | [kernel](triton_eval_cases/case_97/kernel.py) · [合同](triton_eval_cases/case_97/problem.txt) | [记录](traces_glm/case_97/) |
| `case_98` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / expm1 range reduction | [kernel](triton_eval_cases/case_98/kernel.py) · [合同](triton_eval_cases/case_98/problem.txt) | [记录](traces_glm/case_98/) |
| `case_99` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / asinh branch approximation | [kernel](triton_eval_cases/case_99/kernel.py) · [合同](triton_eval_cases/case_99/problem.txt) | [记录](traces_glm/case_99/) |
| `case_100` | 当前 · 成功案例：已观察到工具流程增益，单独保留 | real_kernel_challenges / asinh branch approximation | [kernel](triton_eval_cases/case_100/kernel.py) · [合同](triton_eval_cases/case_100/problem.txt) | [记录](traces_glm/case_100/) |
| `case_101` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Cube-root refinement | [kernel](triton_eval_cases/case_101/kernel.py) · [合同](triton_eval_cases/case_101/problem.txt) | [记录](traces_glm/case_101/) |
| `case_102` | 当前 · 未体现增益，保留待归普通集合 | real_kernel_challenges / Cube-root refinement | [kernel](triton_eval_cases/case_102/kernel.py) · [合同](triton_eval_cases/case_102/problem.txt) | [记录](traces_glm/case_102/) |
| `case_103` | 已归档 | benchmark_fn_fp / FP32 原子累加的逐位复现 | [kernel](archive/input_specific/triton_eval_cases/case_103/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_103/problem.txt) | [记录](archive/input_specific/traces_glm/case_103/) |
| `case_104` | 已归档 | benchmark_fn_fp / 整数网格上原子累加的精确性 | [kernel](archive/input_specific/triton_eval_cases/case_104/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_104/problem.txt) | [记录](archive/input_specific/traces_glm/case_104/) |
| `case_105` | 已归档 | single_call_vs_tools_challenges / 两路量化误差相加与抵消 | [kernel](archive/input_specific/triton_eval_cases/case_105/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_105/problem.txt) | [记录](archive/input_specific/traces_glm/case_105/) |
| `case_106` | 已归档 | single_call_vs_tools_challenges / 两路量化误差相加与抵消 | [kernel](archive/input_specific/triton_eval_cases/case_106/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_106/problem.txt) | [记录](archive/input_specific/traces_glm/case_106/) |
| `case_107` | 已归档 | single_call_vs_tools_challenges / Attention logits 量化误差 | [kernel](archive/input_specific/triton_eval_cases/case_107/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_107/problem.txt) | [记录](archive/input_specific/traces_glm/case_107/) |
| `case_108` | 已归档 | single_call_vs_tools_challenges / Attention logits 量化误差 | [kernel](archive/input_specific/triton_eval_cases/case_108/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_108/problem.txt) | [记录](archive/input_specific/traces_glm/case_108/) |
| `case_109` | 已归档 | single_call_vs_tools_challenges / FP16 递推舍入误差传播 | [kernel](archive/input_specific/triton_eval_cases/case_109/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_109/problem.txt) | [记录](archive/input_specific/traces_glm/case_109/) |
| `case_110` | 已归档 | single_call_vs_tools_challenges / FP16 递推舍入误差传播 | [kernel](archive/input_specific/triton_eval_cases/case_110/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_110/problem.txt) | [记录](archive/input_specific/traces_glm/case_110/) |
| `case_111` | 已归档 | single_call_vs_tools_challenges / FP32 求和顺序 | [kernel](archive/input_specific/triton_eval_cases/case_111/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_111/problem.txt) | [记录](archive/input_specific/traces_glm/case_111/) |
| `case_112` | 已归档 | single_call_vs_tools_challenges / FP32 求和顺序 | [kernel](archive/input_specific/triton_eval_cases/case_112/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_112/problem.txt) | [记录](archive/input_specific/traces_glm/case_112/) |
| `case_113` | 已归档 | single_call_vs_tools_challenges / LayerNorm 方差相消 | [kernel](archive/input_specific/triton_eval_cases/case_113/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_113/problem.txt) | [记录](archive/input_specific/traces_glm/case_113/) |
| `case_114` | 已归档 | single_call_vs_tools_challenges / LayerNorm 方差相消 | [kernel](archive/input_specific/triton_eval_cases/case_114/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_114/problem.txt) | [记录](archive/input_specific/traces_glm/case_114/) |
| `case_115` | 已归档 | single_call_vs_tools_challenges / 固定步数线性求解 | [kernel](archive/input_specific/triton_eval_cases/case_115/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_115/problem.txt) | [记录](archive/input_specific/traces_glm/case_115/) |
| `case_116` | 已归档 | single_call_vs_tools_challenges / 固定步数线性求解 | [kernel](archive/input_specific/triton_eval_cases/case_116/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_116/problem.txt) | [记录](archive/input_specific/traces_glm/case_116/) |
| `case_117` | 已归档 | single_call_vs_tools_challenges / Horner 多项式舍入 | [kernel](archive/input_specific/triton_eval_cases/case_117/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_117/problem.txt) | [记录](archive/input_specific/traces_glm/case_117/) |
| `case_118` | 已归档 | single_call_vs_tools_challenges / Horner 多项式舍入 | [kernel](archive/input_specific/triton_eval_cases/case_118/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_118/problem.txt) | [记录](archive/input_specific/traces_glm/case_118/) |
| `case_119` | 已归档 | single_call_vs_tools_challenges / 近共线投影与归一化 | [kernel](archive/input_specific/triton_eval_cases/case_119/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_119/problem.txt) | [记录](archive/input_specific/traces_glm/case_119/) |
| `case_120` | 已归档 | single_call_vs_tools_challenges / 近共线投影与归一化 | [kernel](archive/input_specific/triton_eval_cases/case_120/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_120/problem.txt) | [记录](archive/input_specific/traces_glm/case_120/) |
| `case_121` | 已归档 | single_call_vs_tools_challenges / 量化最近邻与路由 | [kernel](archive/input_specific/triton_eval_cases/case_121/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_121/problem.txt) | [记录](archive/input_specific/traces_glm/case_121/) |
| `case_122` | 已归档 | single_call_vs_tools_challenges / 量化最近邻与路由 | [kernel](archive/input_specific/triton_eval_cases/case_122/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_122/problem.txt) | [记录](archive/input_specific/traces_glm/case_122/) |
| `case_123` | 已归档 | single_call_vs_tools_challenges / 固定网格积分欠采样 | [kernel](archive/input_specific/triton_eval_cases/case_123/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_123/problem.txt) | [记录](archive/input_specific/traces_glm/case_123/) |
| `case_124` | 已归档 | single_call_vs_tools_challenges / 固定网格积分欠采样 | [kernel](archive/input_specific/triton_eval_cases/case_124/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_124/problem.txt) | [记录](archive/input_specific/traces_glm/case_124/) |
| `case_125` | 已归档 | single_call_vs_tools_challenges / Fourier 频带截断 | [kernel](archive/input_specific/triton_eval_cases/case_125/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_125/problem.txt) | [记录](archive/input_specific/traces_glm/case_125/) |
| `case_126` | 已归档 | single_call_vs_tools_challenges / Fourier 频带截断 | [kernel](archive/input_specific/triton_eval_cases/case_126/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_126/problem.txt) | [记录](archive/input_specific/traces_glm/case_126/) |
| `case_127` | 已归档 | single_call_vs_tools_challenges / 病态矩阵 logdet | [kernel](archive/input_specific/triton_eval_cases/case_127/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_127/problem.txt) | [记录](archive/input_specific/traces_glm/case_127/) |
| `case_128` | 已归档 | single_call_vs_tools_challenges / 病态矩阵 logdet | [kernel](archive/input_specific/triton_eval_cases/case_128/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_128/problem.txt) | [记录](archive/input_specific/traces_glm/case_128/) |
| `case_129` | 已归档 | single_call_vs_tools_challenges / RBF 距离展开相消 | [kernel](archive/input_specific/triton_eval_cases/case_129/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_129/problem.txt) | [记录](archive/input_specific/traces_glm/case_129/) |
| `case_130` | 已归档 | single_call_vs_tools_challenges / RBF 距离展开相消 | [kernel](archive/input_specific/triton_eval_cases/case_130/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_130/problem.txt) | [记录](archive/input_specific/traces_glm/case_130/) |
| `case_131` | 已归档 | solo_vs_debate_challenges / 协方差回归的参考独立性 | [kernel](archive/input_specific/triton_eval_cases/case_131/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_131/problem.txt) | [记录](archive/input_specific/traces_glm/case_131/) |
| `case_132` | 已归档 | solo_vs_debate_challenges / 协方差回归的参考独立性 | [kernel](archive/input_specific/triton_eval_cases/case_132/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_132/problem.txt) | [记录](archive/input_specific/traces_glm/case_132/) |
| `case_133` | 已归档 | solo_vs_debate_challenges / 缓存状态序列覆盖 | [kernel](archive/input_specific/triton_eval_cases/case_133/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_133/problem.txt) | [记录](archive/input_specific/traces_glm/case_133/) |
| `case_134` | 已归档 | solo_vs_debate_challenges / 缓存状态序列覆盖 | [kernel](archive/input_specific/triton_eval_cases/case_134/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_134/problem.txt) | [记录](archive/input_specific/traces_glm/case_134/) |
| `case_135` | 已归档 | solo_vs_debate_challenges / 四阶联合分布 | [kernel](archive/input_specific/triton_eval_cases/case_135/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_135/problem.txt) | [记录](archive/input_specific/traces_glm/case_135/) |
| `case_136` | 已归档 | solo_vs_debate_challenges / 四阶联合分布 | [kernel](archive/input_specific/triton_eval_cases/case_136/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_136/problem.txt) | [记录](archive/input_specific/traces_glm/case_136/) |
| `case_137` | 已归档 | solo_vs_debate_challenges / 局部匹配与全局共同变换 | [kernel](archive/input_specific/triton_eval_cases/case_137/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_137/problem.txt) | [记录](archive/input_specific/traces_glm/case_137/) |
| `case_138` | 已归档 | solo_vs_debate_challenges / 局部匹配与全局共同变换 | [kernel](archive/input_specific/triton_eval_cases/case_138/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_138/problem.txt) | [记录](archive/input_specific/traces_glm/case_138/) |
| `case_139` | 已归档 | solo_vs_debate_challenges / 历史输出被后续更新覆盖 | [kernel](archive/input_specific/triton_eval_cases/case_139/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_139/problem.txt) | [记录](archive/input_specific/traces_glm/case_139/) |
| `case_140` | 已归档 | solo_vs_debate_challenges / 历史输出被后续更新覆盖 | [kernel](archive/input_specific/triton_eval_cases/case_140/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_140/problem.txt) | [记录](archive/input_specific/traces_glm/case_140/) |
| `case_141` | 已归档 | solo_vs_debate_challenges / 连续输入域的最坏情况 | [kernel](archive/input_specific/triton_eval_cases/case_141/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_141/problem.txt) | [记录](archive/input_specific/traces_glm/case_141/) |
| `case_142` | 已归档 | solo_vs_debate_challenges / 连续输入域的最坏情况 | [kernel](archive/input_specific/triton_eval_cases/case_142/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_142/problem.txt) | [记录](archive/input_specific/traces_glm/case_142/) |
| `case_143` | 已归档 | solo_vs_debate_challenges / FP64 求和参考失真 | [kernel](archive/input_specific/triton_eval_cases/case_143/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_143/problem.txt) | [记录](archive/input_specific/traces_glm/case_143/) |
| `case_144` | 已归档 | solo_vs_debate_challenges / FP64 求和参考失真 | [kernel](archive/input_specific/triton_eval_cases/case_144/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_144/problem.txt) | [记录](archive/input_specific/traces_glm/case_144/) |
| `case_145` | 已归档 | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](archive/input_specific/triton_eval_cases/case_145/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_145/problem.txt) | [记录](archive/input_specific/traces_glm/case_145/) |
| `case_146` | 已归档 | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](archive/input_specific/triton_eval_cases/case_146/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_146/problem.txt) | [记录](archive/input_specific/traces_glm/case_146/) |
| `case_147` | 已归档 | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](archive/input_specific/triton_eval_cases/case_147/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_147/problem.txt) | [记录](archive/input_specific/traces_glm/case_147/) |
| `case_148` | 已归档 | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](archive/input_specific/triton_eval_cases/case_148/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_148/problem.txt) | [记录](archive/input_specific/traces_glm/case_148/) |
| `case_149` | 已归档 | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](archive/input_specific/triton_eval_cases/case_149/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_149/problem.txt) | [记录](archive/input_specific/traces_glm/case_149/) |
| `case_150` | 已归档 | solo_vs_debate_challenges / 求和参考失真：新输入确认 | [kernel](archive/input_specific/triton_eval_cases/case_150/kernel.py) · [合同](archive/input_specific/triton_eval_cases/case_150/problem.txt) | [记录](archive/input_specific/traces_glm/case_150/) |
| `case_151` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_151/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_151/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_151/) |
| `case_152` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_152/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_152/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_152/) |
| `case_153` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_153/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_153/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_153/) |
| `case_154` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_154/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_154/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_154/) |
| `case_155` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_155/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_155/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_155/) |
| `case_156` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_156/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_156/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_156/) |
| `case_157` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_157/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_157/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_157/) |
| `case_158` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_158/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_158/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_158/) |
| `case_159` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_159/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_159/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_159/) |
| `case_160` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_160/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_160/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_160/) |
| `case_161` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_161/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_161/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_161/) |
| `case_162` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_162/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_162/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_162/) |
| `case_163` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_163/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_163/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_163/) |
| `case_164` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_164/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_164/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_164/) |
| `case_165` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_165/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_165/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_165/) |
| `case_166` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_166/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_166/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_166/) |
| `case_167` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_167/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_167/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_167/) |
| `case_168` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_168/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_168/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_168/) |
| `case_169` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_169/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_169/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_169/) |
| `case_170` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_170/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_170/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_170/) |
| `case_171` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_171/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_171/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_171/) |
| `case_172` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_172/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_172/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_172/) |
| `case_173` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_173/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_173/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_173/) |
| `case_174` | 已归档 | numerical_pilot / 早期数值 pilot： | [kernel](archive/numerical_pilot/eval_cases/case_174/kernel.py) · [合同](archive/numerical_pilot/eval_cases/case_174/problem.txt) | [记录](archive/numerical_pilot/traces_glm/case_174/) |

## 如何读 traces

`traces_glm/<case>/<arm>/<trial>/` 中，`single_call` 是无工具单次调用，`solo` 是单 agent 加工具，`debate` 是多角色加工具。各组批次目录统一为 `r1`、`r2` 等运行序号；原批次名保留在 `trace_meta.json` 的 `original_trial` 中。同名 `r1` 不代表模型、预算或实验配置相同。
[GLM 逐次运行索引](traces_glm/INDEX.md) 保留不同配置、复测与失败尝试；题库编号相同不代表三组都已跑齐。
目录使用当前编号；历史请求、回复、metadata 和 probe 代码保留原始字节。读取程序根据当前目录和注册表核对历史记录的身份。
归档案例的公开题目与 traces 按上表移入对应归档目录，历史来源字段和原始调用内容不改写。
早期数值 pilot 的题目和历史记录保存在 archive/numerical_pilot，已从当前 GLM 索引与默认评分移出。

## 文件清理说明

各来源组的 `private_data/` 保存答案、搜索过程和 GPU 真值校验；这些是实验可复现性依据。归档组的资料留在归档目录。
当前题目统一在 `triton_eval_cases/`，构造、校验与报告程序统一在 `eval_scripts/` 的对应目录；已归档的公开题目不再留在当前目录。
零散实验说明已合并进各组 README；旧逻辑数据集名只用于兼容历史 traces 和旧命令。可从原始 traces 重建的 scoreboard JSON 已清理；报告脚本默认更新 README 中的结果表，如需机器可读汇总可显式请求 JSON。原始模型调用与 GPU 记录不属于可重建汇总。

本页由 `eval_scripts/index_cases.py` 根据 `case_map.json` 生成。
