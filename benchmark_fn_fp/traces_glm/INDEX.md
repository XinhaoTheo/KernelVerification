# GLM trace index

Current benchmark GLM runs are listed together by case, arm and trial. API provenance is retained in trace metadata. Historical tool traces do not have raw API payloads; new runs include `llm_calls/`.

See [案例总索引](../CASE_INDEX.md) for numeric ranges, original IDs and source kernels. Historical trace payloads keep their original IDs; the current directory and registry determine the case.

每个 case/arm 下按历史时间顺序编号为 `r1`、`r2`……；不同案例的同名 rN 不代表同一实验批次。原始批次保存在 `trace_meta.json` 的 `original_trial`，统计继续按批次、模型和预算配置分组。

早期 pilot 的 24 题、72 条 GLM trials 已移入[归档](../archive/numerical_pilot/README.md)，不计入下方当前题库覆盖。历史记录保留；归档不表示重跑或改判。

## 实验后的保留分组

这是后续整理状态，原始题库来源与全部实验统计保留。单次观察到增益不代表稳定优势；待归普通集合的案例目前仍保留在原来源组。

| 分组 | Case / traces |
|---|---|
| 成功案例：已观察到工具流程增益，单独保留 | [case_109](case_109/)、[case_114](case_114/) |
| 未体现增益，保留待归普通集合 | [case_106](case_106/)、[case_107](case_107/)、[case_108](case_108/)、[case_110](case_110/)、[case_111](case_111/)、[case_112](case_112/)、[case_113](case_113/)、[case_115](case_115/) |

## 全题库三组覆盖

范围：case_map.json 中全部 90 个活跃案例，共 270 个 case-arm 槽位。
完成包含正确、错误和 needs_more_evidence（弃答）；截断、无最终判断、运行错误和运行中均不算完成。错误答案和有效弃答保留，不因结果不理想而补跑。

| 实验组 | 已完成 | 从未运行 | 失败/未完成 | 运行中 | 未完成合计 |
|---|---:|---:|---:|---:|---:|
| single_call | 90 | 0 | 0 | 0 | 0 |
| solo | 90 | 0 | 0 | 0 | 0 |
| debate | 89 | 0 | 1 | 0 | 1 |

完成记录按最早的 created_at 选取；缺少日期的旧记录优先，并以迁移前路径稳定排序。不按正确性或置信度筛选；下方索引保留当前题库的全部尝试。
本页是覆盖清单，不是统一协议的准确率对照：历史记录包含 OpenRouter 的 `z-ai/glm-5.3-flash` 与 Fireworks 的 `accounts/fireworks/models/glm-5p3`，模型、推理配置、预算和运行器版本可能不同。
选用记录中 16 条没有可确认的完整原始 API capture；历史缺失保留披露，不为补日志重跑。补跑前审计有 16 条已完成旧记录属于这种情况。

| Case | 题库 | single_call | solo | debate |
|---|---|---|---|---|
| case_01 | benchmark_fn_fp | [r1](case_01/single_call/r1/transcript.md) · trust / 错误 | [r1](case_01/solo/r1/transcript.md) · trust / 错误 | [r1](case_01/debate/r1/transcript.md) · reject / 正确 |
| case_02 | benchmark_fn_fp | [r1](case_02/single_call/r1/transcript.md) · trust / 正确 | [r1](case_02/solo/r1/transcript.md) · trust / 正确 | [r1](case_02/debate/r1/transcript.md) · trust / 正确 |
| case_04 | benchmark_fn_fp | [r1](case_04/single_call/r1/transcript.md) · trust / 正确 | [r1](case_04/solo/r1/transcript.md) · trust / 正确 | [r1](case_04/debate/r1/transcript.md) · trust / 正确 |
| case_05 | benchmark_fn_fp | [r1](case_05/single_call/r1/transcript.md) · trust / 正确 | [r1](case_05/solo/r1/transcript.md) · trust / 正确 | [r1](case_05/debate/r1/transcript.md) · trust / 正确 |
| case_06 | benchmark_fn_fp | [r1](case_06/single_call/r1/transcript.md) · reject / 正确 | [r1](case_06/solo/r1/transcript.md) · reject / 正确 | [r1](case_06/debate/r1/transcript.md) · reject / 正确 |
| case_07 | benchmark_fn_fp | [r1](case_07/single_call/r1/transcript.md) · trust / 正确 | [r1](case_07/solo/r1/transcript.md) · trust / 正确 | [r1](case_07/debate/r1/transcript.md) · needs_more_evidence / 弃答 |
| case_08 | benchmark_fn_fp | [r1](case_08/single_call/r1/transcript.md) · trust / 正确 | [r1](case_08/solo/r1/transcript.md) · trust / 正确 | [r1](case_08/debate/r1/transcript.md) · trust / 正确 |
| case_09 | benchmark_fn_fp | [r1](case_09/single_call/r1/transcript.md) · reject / 正确 | [r1](case_09/solo/r1/transcript.md) · reject / 正确 | [r1](case_09/debate/r1/transcript.md) · reject / 正确 |
| case_10 | benchmark_fn_fp | [r1](case_10/single_call/r1/transcript.md) · reject / 正确 | [r1](case_10/solo/r1/transcript.md) · reject / 正确 | [r1](case_10/debate/r1/transcript.md) · reject / 正确 |
| case_11 | benchmark_fn_fp | [r1](case_11/single_call/r1/transcript.md) · trust / 正确 | [r2](case_11/solo/r2/transcript.md) · trust / 正确 | [r1](case_11/debate/r1/transcript.md) · trust / 正确 |
| case_12 | benchmark_fn_fp | [r1](case_12/single_call/r1/transcript.md) · reject / 正确 | [r1](case_12/solo/r1/transcript.md) · reject / 正确 | [r1](case_12/debate/r1/transcript.md) · reject / 正确 |
| case_13 | benchmark_fn_fp | [r1](case_13/single_call/r1/transcript.md) · reject / 正确 | [r1](case_13/solo/r1/transcript.md) · reject / 正确 | [r1](case_13/debate/r1/transcript.md) · reject / 正确 |
| case_14 | benchmark_fn_fp | [r1](case_14/single_call/r1/transcript.md) · reject / 正确 | [r1](case_14/solo/r1/transcript.md) · reject / 正确 | [r1](case_14/debate/r1/transcript.md) · reject / 正确 |
| case_15 | benchmark_fn_fp | [r1](case_15/single_call/r1/transcript.md) · reject / 正确 | [r1](case_15/solo/r1/transcript.md) · reject / 正确 | [r1](case_15/debate/r1/transcript.md) · reject / 正确 |
| case_16 | benchmark_fn_fp | [r1](case_16/single_call/r1/transcript.md) · reject / 正确 | [r1](case_16/solo/r1/transcript.md) · reject / 正确 | [r1](case_16/debate/r1/transcript.md) · reject / 正确 |
| case_17 | benchmark_fn_fp | [r1](case_17/single_call/r1/transcript.md) · reject / 正确 | [r1](case_17/solo/r1/transcript.md) · reject / 正确 | [r1](case_17/debate/r1/transcript.md) · reject / 正确 |
| case_18 | benchmark_fn_fp | [r1](case_18/single_call/r1/transcript.md) · reject / 正确 | [r1](case_18/solo/r1/transcript.md) · reject / 正确 | [r1](case_18/debate/r1/transcript.md) · reject / 正确 |
| case_19 | benchmark_fn_fp | [r1](case_19/single_call/r1/transcript.md) · reject / 正确 | [r1](case_19/solo/r1/transcript.md) · reject / 正确 | [r1](case_19/debate/r1/transcript.md) · reject / 正确 |
| case_20 | benchmark_fn_fp | [r1](case_20/single_call/r1/transcript.md) · reject / 正确 | [r1](case_20/solo/r1/transcript.md) · reject / 正确 | [r1](case_20/debate/r1/transcript.md) · needs_more_evidence / 弃答 |
| case_21 | benchmark_fn_fp | [r1](case_21/single_call/r1/transcript.md) · reject / 正确 | [r1](case_21/solo/r1/transcript.md) · reject / 正确 | [r1](case_21/debate/r1/transcript.md) · reject / 正确 |
| case_22 | benchmark_fn_fp | [r1](case_22/single_call/r1/transcript.md) · trust / 正确 | [r1](case_22/solo/r1/transcript.md) · trust / 正确 | [r1](case_22/debate/r1/transcript.md) · trust / 正确 |
| case_23 | benchmark_fn_fp | [r1](case_23/single_call/r1/transcript.md) · trust / 正确 | [r1](case_23/solo/r1/transcript.md) · trust / 正确 | [r1](case_23/debate/r1/transcript.md) · reject / 错误 |
| case_24 | benchmark_fn_fp | [r1](case_24/single_call/r1/transcript.md) · trust / 正确 | [r1](case_24/solo/r1/transcript.md) · trust / 正确 | [r1](case_24/debate/r1/transcript.md) · trust / 正确 |
| case_25 | benchmark_fn_fp | [r1](case_25/single_call/r1/transcript.md) · reject / 正确 | [r1](case_25/solo/r1/transcript.md) · reject / 正确 | [r1](case_25/debate/r1/transcript.md) · reject / 正确 |
| case_26 | benchmark_fn_fp | [r1](case_26/single_call/r1/transcript.md) · trust / 正确 | [r1](case_26/solo/r1/transcript.md) · trust / 正确 | [r1](case_26/debate/r1/transcript.md) · needs_more_evidence / 弃答 |
| case_27 | benchmark_fn_fp | [r1](case_27/single_call/r1/transcript.md) · reject / 正确 | [r1](case_27/solo/r1/transcript.md) · reject / 正确 | [r1](case_27/debate/r1/transcript.md) · reject / 正确 |
| case_28 | benchmark_fn_fp | [r1](case_28/single_call/r1/transcript.md) · reject / 正确 | [r1](case_28/solo/r1/transcript.md) · reject / 正确 | [r1](case_28/debate/r1/transcript.md) · reject / 正确 |
| case_29 | benchmark_fn_fp | [r1](case_29/single_call/r1/transcript.md) · reject / 错误 | [r1](case_29/solo/r1/transcript.md) · reject / 错误 | [r1](case_29/debate/r1/transcript.md) · reject / 错误 |
| case_30 | benchmark_fn_fp | [r1](case_30/single_call/r1/transcript.md) · reject / 正确 | [r1](case_30/solo/r1/transcript.md) · reject / 正确 | [r1](case_30/debate/r1/transcript.md) · reject / 正确 |
| case_31 | benchmark_fn_fp | [r1](case_31/single_call/r1/transcript.md) · reject / 正确 | [r1](case_31/solo/r1/transcript.md) · reject / 正确 | [r1](case_31/debate/r1/transcript.md) · reject / 正确 |
| case_32 | benchmark_fn_fp | [r1](case_32/single_call/r1/transcript.md) · reject / 正确 | [r1](case_32/solo/r1/transcript.md) · reject / 正确 | [r1](case_32/debate/r1/transcript.md) · reject / 正确 |
| case_33 | benchmark_fn_fp | [r1](case_33/single_call/r1/transcript.md) · reject / 正确 | [r1](case_33/solo/r1/transcript.md) · reject / 正确 | [r1](case_33/debate/r1/transcript.md) · reject / 正确 |
| case_34 | benchmark_fn_fp | [r1](case_34/single_call/r1/transcript.md) · reject / 正确 | [r1](case_34/solo/r1/transcript.md) · reject / 正确 | [r1](case_34/debate/r1/transcript.md) · reject / 正确 |
| case_35 | benchmark_fn_fp | [r1](case_35/single_call/r1/transcript.md) · reject / 错误 | [r1](case_35/solo/r1/transcript.md) · trust / 正确 | [r1](case_35/debate/r1/transcript.md) · trust / 正确 |
| case_36 | single_call_vs_tools_challenges | [r1](case_36/single_call/r1/transcript.md) · reject / 正确 | [r1](case_36/solo/r1/transcript.md) · reject / 正确 | [r1](case_36/debate/r1/transcript.md) · reject / 正确 |
| case_37 | single_call_vs_tools_challenges | [r2](case_37/single_call/r2/transcript.md) · reject / 错误 | [r1](case_37/solo/r1/transcript.md) · trust / 正确 | [r1](case_37/debate/r1/transcript.md) · trust / 正确 |
| case_38 | single_call_vs_tools_challenges | [r5](case_38/single_call/r5/transcript.md) · reject / 错误 | [r1](case_38/solo/r1/transcript.md) · trust / 正确 | [r2](case_38/debate/r2/transcript.md) · trust / 正确 |
| case_39 | single_call_vs_tools_challenges | [r1](case_39/single_call/r1/transcript.md) · reject / 正确 | [r2](case_39/solo/r2/transcript.md) · reject / 正确 | [r2](case_39/debate/r2/transcript.md) · reject / 正确 |
| case_40 | single_call_vs_tools_challenges | [r3](case_40/single_call/r3/transcript.md) · trust / 正确 | [r2](case_40/solo/r2/transcript.md) · trust / 正确 | [r2](case_40/debate/r2/transcript.md) · trust / 正确 |
| case_41 | single_call_vs_tools_challenges | [r2](case_41/single_call/r2/transcript.md) · trust / 错误 | [r2](case_41/solo/r2/transcript.md) · reject / 正确 | [r2](case_41/debate/r2/transcript.md) · reject / 正确 |
| case_42 | single_call_vs_tools_challenges | [r2](case_42/single_call/r2/transcript.md) · reject / 正确 | [r2](case_42/solo/r2/transcript.md) · reject / 正确 | [r2](case_42/debate/r2/transcript.md) · reject / 正确 |
| case_43 | single_call_vs_tools_challenges | [r2](case_43/single_call/r2/transcript.md) · reject / 错误 | [r2](case_43/solo/r2/transcript.md) · trust / 正确 | [r2](case_43/debate/r2/transcript.md) · trust / 正确 |
| case_44 | single_call_vs_tools_challenges | [r1](case_44/single_call/r1/transcript.md) · reject / 错误 | [r1](case_44/solo/r1/transcript.md) · trust / 正确 | [r1](case_44/debate/r1/transcript.md) · trust / 正确 |
| case_45 | single_call_vs_tools_challenges | [r1](case_45/single_call/r1/transcript.md) · reject / 正确 | [r1](case_45/solo/r1/transcript.md) · reject / 正确 | [r1](case_45/debate/r1/transcript.md) · reject / 正确 |
| case_46 | single_call_vs_tools_challenges | [r1](case_46/single_call/r1/transcript.md) · reject / 错误 | [r1](case_46/solo/r1/transcript.md) · trust / 正确 | [r1](case_46/debate/r1/transcript.md) · trust / 正确 |
| case_47 | single_call_vs_tools_challenges | [r1](case_47/single_call/r1/transcript.md) · reject / 正确 | [r1](case_47/solo/r1/transcript.md) · reject / 正确 | [r1](case_47/debate/r1/transcript.md) · reject / 正确 |
| case_48 | single_call_vs_tools_challenges | [r1](case_48/single_call/r1/transcript.md) · reject / 错误 | [r1](case_48/solo/r1/transcript.md) · trust / 正确 | [r1](case_48/debate/r1/transcript.md) · trust / 正确 |
| case_49 | single_call_vs_tools_challenges | [r1](case_49/single_call/r1/transcript.md) · reject / 正确 | [r1](case_49/solo/r1/transcript.md) · reject / 正确 | [r1](case_49/debate/r1/transcript.md) · reject / 正确 |
| case_50 | single_call_vs_tools_challenges | [r1](case_50/single_call/r1/transcript.md) · reject / 错误 | [r1](case_50/solo/r1/transcript.md) · trust / 正确 | [r1](case_50/debate/r1/transcript.md) · trust / 正确 |
| case_51 | single_call_vs_tools_challenges | [r1](case_51/single_call/r1/transcript.md) · reject / 正确 | [r1](case_51/solo/r1/transcript.md) · reject / 正确 | [r1](case_51/debate/r1/transcript.md) · reject / 正确 |
| case_52 | single_call_vs_tools_challenges | [r1](case_52/single_call/r1/transcript.md) · reject / 错误 | [r1](case_52/solo/r1/transcript.md) · trust / 正确 | [r1](case_52/debate/r1/transcript.md) · trust / 正确 |
| case_53 | single_call_vs_tools_challenges | [r1](case_53/single_call/r1/transcript.md) · reject / 正确 | [r1](case_53/solo/r1/transcript.md) · reject / 正确 | [r1](case_53/debate/r1/transcript.md) · reject / 正确 |
| case_54 | single_call_vs_tools_challenges | [r2](case_54/single_call/r2/transcript.md) · reject / 错误 | [r1](case_54/solo/r1/transcript.md) · trust / 正确 | [r1](case_54/debate/r1/transcript.md) · trust / 正确 |
| case_55 | single_call_vs_tools_challenges | [r2](case_55/single_call/r2/transcript.md) · trust / 错误 | [r1](case_55/solo/r1/transcript.md) · reject / 正确 | [r1](case_55/debate/r1/transcript.md) · reject / 正确 |
| case_56 | single_call_vs_tools_challenges | [r1](case_56/single_call/r1/transcript.md) · reject / 错误 | [r1](case_56/solo/r1/transcript.md) · trust / 正确 | [r1](case_56/debate/r1/transcript.md) · trust / 正确 |
| case_57 | single_call_vs_tools_challenges | [r1](case_57/single_call/r1/transcript.md) · needs_more_evidence / 弃答 | [r1](case_57/solo/r1/transcript.md) · reject / 正确 | [r1](case_57/debate/r1/transcript.md) · reject / 正确 |
| case_58 | single_call_vs_tools_challenges | [r1](case_58/single_call/r1/transcript.md) · trust / 正确 | [r1](case_58/solo/r1/transcript.md) · trust / 正确 | [r1](case_58/debate/r1/transcript.md) · trust / 正确 |
| case_59 | single_call_vs_tools_challenges | [r1](case_59/single_call/r1/transcript.md) · trust / 错误 | [r1](case_59/solo/r1/transcript.md) · reject / 正确 | [r1](case_59/debate/r1/transcript.md) · reject / 正确 |
| case_60 | single_call_vs_tools_challenges | [r1](case_60/single_call/r1/transcript.md) · reject / 错误 | [r1](case_60/solo/r1/transcript.md) · trust / 正确 | [r1](case_60/debate/r1/transcript.md) · trust / 正确 |
| case_61 | single_call_vs_tools_challenges | [r1](case_61/single_call/r1/transcript.md) · trust / 错误 | [r1](case_61/solo/r1/transcript.md) · reject / 正确 | [r1](case_61/debate/r1/transcript.md) · reject / 正确 |
| case_62 | solo_vs_debate_challenges | [r1](case_62/single_call/r1/transcript.md) · trust / 正确 | [r1](case_62/solo/r1/transcript.md) · trust / 正确 | [r1](case_62/debate/r1/transcript.md) · trust / 正确 |
| case_63 | solo_vs_debate_challenges | [r1](case_63/single_call/r1/transcript.md) · reject / 正确 | [r1](case_63/solo/r1/transcript.md) · reject / 正确 | [r1](case_63/debate/r1/transcript.md) · reject / 正确 |
| case_64 | solo_vs_debate_challenges | [r1](case_64/single_call/r1/transcript.md) · reject / 错误 | [r1](case_64/solo/r1/transcript.md) · trust / 正确 | [r1](case_64/debate/r1/transcript.md) · trust / 正确 |
| case_65 | solo_vs_debate_challenges | [r1](case_65/single_call/r1/transcript.md) · reject / 正确 | [r1](case_65/solo/r1/transcript.md) · reject / 正确 | [r1](case_65/debate/r1/transcript.md) · reject / 正确 |
| case_66 | solo_vs_debate_challenges | [r1](case_66/single_call/r1/transcript.md) · needs_more_evidence / 弃答 | [r1](case_66/solo/r1/transcript.md) · trust / 正确 | [r1](case_66/debate/r1/transcript.md) · trust / 正确 |
| case_67 | solo_vs_debate_challenges | [r1](case_67/single_call/r1/transcript.md) · trust / 错误 | [r1](case_67/solo/r1/transcript.md) · reject / 正确 | [r1](case_67/debate/r1/transcript.md) · reject / 正确 |
| case_68 | solo_vs_debate_challenges | [r1](case_68/single_call/r1/transcript.md) · trust / 正确 | [r1](case_68/solo/r1/transcript.md) · trust / 正确 | [r1](case_68/debate/r1/transcript.md) · trust / 正确 |
| case_69 | solo_vs_debate_challenges | [r1](case_69/single_call/r1/transcript.md) · reject / 正确 | [r1](case_69/solo/r1/transcript.md) · reject / 正确 | [r1](case_69/debate/r1/transcript.md) · reject / 正确 |
| case_70 | solo_vs_debate_challenges | [r1](case_70/single_call/r1/transcript.md) · reject / 错误 | [r1](case_70/solo/r1/transcript.md) · trust / 正确 | [r1](case_70/debate/r1/transcript.md) · trust / 正确 |
| case_71 | solo_vs_debate_challenges | [r1](case_71/single_call/r1/transcript.md) · reject / 正确 | [r1](case_71/solo/r1/transcript.md) · reject / 正确 | [r1](case_71/debate/r1/transcript.md) · reject / 正确 |
| case_72 | solo_vs_debate_challenges | [r1](case_72/single_call/r1/transcript.md) · reject / 错误 | [r1](case_72/solo/r1/transcript.md) · trust / 正确 | [r1](case_72/debate/r1/transcript.md) · trust / 正确 |
| case_73 | solo_vs_debate_challenges | [r1](case_73/single_call/r1/transcript.md) · reject / 正确 | [r1](case_73/solo/r1/transcript.md) · reject / 正确 | [r1](case_73/debate/r1/transcript.md) · reject / 正确 |
| case_74 | solo_vs_debate_challenges | [r1](case_74/single_call/r1/transcript.md) · reject / 错误 | [r1](case_74/solo/r1/transcript.md) · reject / 错误 | [r1](case_74/debate/r1/transcript.md) · trust / 正确 |
| case_75 | solo_vs_debate_challenges | [r1](case_75/single_call/r1/transcript.md) · trust / 错误 | [r1](case_75/solo/r1/transcript.md) · reject / 正确 | [r1](case_75/debate/r1/transcript.md) · needs_more_evidence / 弃答 |
| case_76 | solo_vs_debate_challenges | [r1](case_76/single_call/r1/transcript.md) · reject / 错误 | [r1](case_76/solo/r1/transcript.md) · trust / 正确 | [r1](case_76/debate/r1/transcript.md) · trust / 正确 |
| case_77 | solo_vs_debate_challenges | [r1](case_77/single_call/r1/transcript.md) · trust / 错误 | [r1](case_77/solo/r1/transcript.md) · reject / 正确 | [r1](case_77/debate/r1/transcript.md) · reject / 正确 |
| case_78 | solo_vs_debate_challenges | [r1](case_78/single_call/r1/transcript.md) · trust / 正确 | [r1](case_78/solo/r1/transcript.md) · reject / 错误 | [r1](case_78/debate/r1/transcript.md) · trust / 正确 |
| case_79 | solo_vs_debate_challenges | [r1](case_79/single_call/r1/transcript.md) · trust / 错误 | [r1](case_79/solo/r1/transcript.md) · reject / 正确 | [r1](case_79/debate/r1/transcript.md) · reject / 正确 |
| case_80 | solo_vs_debate_challenges | [r1](case_80/single_call/r1/transcript.md) · trust / 正确 | [r1](case_80/solo/r1/transcript.md) · trust / 正确 | [r1](case_80/debate/r1/transcript.md) · trust / 正确 |
| case_81 | solo_vs_debate_challenges | [r1](case_81/single_call/r1/transcript.md) · trust / 错误 | [r1](case_81/solo/r1/transcript.md) · reject / 正确 | [r1](case_81/debate/r1/transcript.md) · reject / 正确 |
| case_106 | real_kernel_challenges | [r1](case_106/single_call/r1/transcript.md) · reject / 正确 | [r1](case_106/solo/r1/transcript.md) · reject / 正确 | [r1](case_106/debate/r1/transcript.md) · reject / 正确 |
| case_107 | real_kernel_challenges | [r1](case_107/single_call/r1/transcript.md) · trust / 错误 | [r1](case_107/solo/r1/transcript.md) · trust / 错误 | [r1](case_107/debate/r1/transcript.md) · trust / 错误 |
| case_108 | real_kernel_challenges | [r1](case_108/single_call/r1/transcript.md) · trust / 错误 | [r2](case_108/solo/r2/transcript.md) · trust / 错误 | [r2](case_108/debate/r2/transcript.md) · trust / 错误 |
| case_109 | real_kernel_challenges | [r1](case_109/single_call/r1/transcript.md) · trust / 错误 | [r2](case_109/solo/r2/transcript.md) · reject / 正确 | 失败/未完成 |
| case_110 | real_kernel_challenges | [r1](case_110/single_call/r1/transcript.md) · trust / 正确 | [r1](case_110/solo/r1/transcript.md) · trust / 正确 | [r1](case_110/debate/r1/transcript.md) · trust / 正确 |
| case_111 | real_kernel_challenges | [r1](case_111/single_call/r1/transcript.md) · trust / 正确 | [r1](case_111/solo/r1/transcript.md) · trust / 正确 | [r1](case_111/debate/r1/transcript.md) · trust / 正确 |
| case_112 | real_kernel_challenges | [r1](case_112/single_call/r1/transcript.md) · trust / 正确 | [r1](case_112/solo/r1/transcript.md) · trust / 正确 | [r1](case_112/debate/r1/transcript.md) · trust / 正确 |
| case_113 | real_kernel_challenges | [r1](case_113/single_call/r1/transcript.md) · reject / 正确 | [r1](case_113/solo/r1/transcript.md) · reject / 正确 | [r1](case_113/debate/r1/transcript.md) · reject / 正确 |
| case_114 | real_kernel_challenges | [r1](case_114/single_call/r1/transcript.md) · trust / 错误 | [r1](case_114/solo/r1/transcript.md) · reject / 正确 | [r1](case_114/debate/r1/transcript.md) · reject / 正确 |
| case_115 | real_kernel_challenges | [r1](case_115/single_call/r1/transcript.md) · trust / 正确 | [r1](case_115/solo/r1/transcript.md) · trust / 正确 | [r1](case_115/debate/r1/transcript.md) · trust / 正确 |

## 全部历史 trials


### 原始 FN/FP benchmark

| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_fn_fp | case_01 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_01/debate/r1/transcript.md) |
| benchmark_fn_fp | case_01 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_01/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_01 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | trust / wrong_verdict | [open](case_01/solo/r1/transcript.md) |
| benchmark_fn_fp | case_02 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_02/debate/r1/transcript.md) |
| benchmark_fn_fp | case_02 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_02/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_02 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_02/solo/r1/transcript.md) |
| benchmark_fn_fp | case_04 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_04/debate/r1/transcript.md) |
| benchmark_fn_fp | case_04 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_04/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_04 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | trust / correct | [open](case_04/solo/r1/transcript.md) |
| benchmark_fn_fp | case_05 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_05/debate/r1/transcript.md) |
| benchmark_fn_fp | case_05 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_05/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_05 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_05/solo/r1/transcript.md) |
| benchmark_fn_fp | case_06 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_06/debate/r1/transcript.md) |
| benchmark_fn_fp | case_06 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_06/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_06 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_06/solo/r1/transcript.md) |
| benchmark_fn_fp | case_07 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_07/debate/r1/transcript.md) |
| benchmark_fn_fp | case_07 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_07/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_07 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_07/solo/r1/transcript.md) |
| benchmark_fn_fp | case_08 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_08/debate/r1/transcript.md) |
| benchmark_fn_fp | case_08 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_08/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_08 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | trust / correct | [open](case_08/solo/r1/transcript.md) |
| benchmark_fn_fp | case_09 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_09/debate/r1/transcript.md) |
| benchmark_fn_fp | case_09 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_09/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_09 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_09/solo/r1/transcript.md) |
| benchmark_fn_fp | case_10 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_10/debate/r1/transcript.md) |
| benchmark_fn_fp | case_10 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_10/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_10 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | reject / correct | [open](case_10/solo/r1/transcript.md) |
| benchmark_fn_fp | case_11 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_11/debate/r1/transcript.md) |
| benchmark_fn_fp | case_11 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_11/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_11 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_11/solo/r1/transcript.md) |
| benchmark_fn_fp | case_11 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_11/solo/r2/transcript.md) |
| benchmark_fn_fp | case_12 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_12/debate/r1/transcript.md) |
| benchmark_fn_fp | case_12 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_12/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_12 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_12/solo/r1/transcript.md) |
| benchmark_fn_fp | case_13 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_13/debate/r1/transcript.md) |
| benchmark_fn_fp | case_13 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_13/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_13 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_13/solo/r1/transcript.md) |
| benchmark_fn_fp | case_14 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_14/debate/r1/transcript.md) |
| benchmark_fn_fp | case_14 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_14/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_14 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_14/solo/r1/transcript.md) |
| benchmark_fn_fp | case_15 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_15/debate/r1/transcript.md) |
| benchmark_fn_fp | case_15 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_15/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_15 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_15/solo/r1/transcript.md) |
| benchmark_fn_fp | case_16 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_16/debate/r1/transcript.md) |
| benchmark_fn_fp | case_16 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_16/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_16 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_16/solo/r1/transcript.md) |
| benchmark_fn_fp | case_17 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_17/debate/r1/transcript.md) |
| benchmark_fn_fp | case_17 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_17/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_17 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_17/solo/r1/transcript.md) |
| benchmark_fn_fp | case_18 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_18/debate/r1/transcript.md) |
| benchmark_fn_fp | case_18 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_18/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_18 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | reject / correct | [open](case_18/solo/r1/transcript.md) |
| benchmark_fn_fp | case_19 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_19/debate/r1/transcript.md) |
| benchmark_fn_fp | case_19 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_19/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_19 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_19/solo/r1/transcript.md) |
| benchmark_fn_fp | case_20 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_20/debate/r1/transcript.md) |
| benchmark_fn_fp | case_20 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_20/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_20 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_20/solo/r1/transcript.md) |
| benchmark_fn_fp | case_21 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_21/debate/r1/transcript.md) |
| benchmark_fn_fp | case_21 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_21/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_21 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_21/solo/r1/transcript.md) |
| benchmark_fn_fp | case_22 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_22/debate/r1/transcript.md) |
| benchmark_fn_fp | case_22 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_22/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_22 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | trust / correct | [open](case_22/solo/r1/transcript.md) |
| benchmark_fn_fp | case_23 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_23/debate/r1/transcript.md) |
| benchmark_fn_fp | case_23 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_23/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_23 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_23/solo/r1/transcript.md) |
| benchmark_fn_fp | case_24 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_24/debate/r1/transcript.md) |
| benchmark_fn_fp | case_24 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_24/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_24 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_24/solo/r1/transcript.md) |
| benchmark_fn_fp | case_25 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_25/debate/r1/transcript.md) |
| benchmark_fn_fp | case_25 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_25/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_25 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_25/solo/r1/transcript.md) |
| benchmark_fn_fp | case_26 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_26/debate/r1/transcript.md) |
| benchmark_fn_fp | case_26 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_26/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_26 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | trust / correct | [open](case_26/solo/r1/transcript.md) |
| benchmark_fn_fp | case_27 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_27/debate/r1/transcript.md) |
| benchmark_fn_fp | case_27 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_27/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_27 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_27/solo/r1/transcript.md) |
| benchmark_fn_fp | case_28 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_28/debate/r1/transcript.md) |
| benchmark_fn_fp | case_28 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_28/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_28 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_28/solo/r1/transcript.md) |
| benchmark_fn_fp | case_29 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_29/debate/r1/transcript.md) |
| benchmark_fn_fp | case_29 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_29/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_29 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | reject / wrong_verdict | [open](case_29/solo/r1/transcript.md) |
| benchmark_fn_fp | case_30 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_30/debate/r1/transcript.md) |
| benchmark_fn_fp | case_30 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_30/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_30 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | reject / correct | [open](case_30/solo/r1/transcript.md) |
| benchmark_fn_fp | case_31 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_31/debate/r1/transcript.md) |
| benchmark_fn_fp | case_31 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_31/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_31 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_31/solo/r1/transcript.md) |
| benchmark_fn_fp | case_32 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_32/debate/r1/transcript.md) |
| benchmark_fn_fp | case_32 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_32/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_32 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_32/solo/r1/transcript.md) |
| benchmark_fn_fp | case_33 | debate | r1 | z-ai/glm-5.3-flash / openrouter | historical | reject / correct | [open](case_33/debate/r1/transcript.md) |
| benchmark_fn_fp | case_33 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_33/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_33 | solo | r1 | z-ai/glm-5.3-flash / openrouter | historical | reject / correct | [open](case_33/solo/r1/transcript.md) |
| benchmark_fn_fp | case_34 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_34/debate/r1/transcript.md) |
| benchmark_fn_fp | case_34 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_34/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_34 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_34/solo/r1/transcript.md) |
| benchmark_fn_fp | case_35 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | trust / correct | [open](case_35/debate/r1/transcript.md) |
| benchmark_fn_fp | case_35 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_35/single_call/r1/transcript.md) |
| benchmark_fn_fp | case_35 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_35/solo/r1/transcript.md) |

### 无工具单次调用 vs 带工具验证

| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| correlation_pair | case_36 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | reject / correct | [open](case_36/debate/r1/transcript.md) |
| correlation_pair | case_36 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | reject / correct | [open](case_36/single_call/r1/transcript.md) |
| correlation_pair | case_36 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | historical | reject / correct | [open](case_36/single_call/r2/transcript.md) |
| correlation_pair | case_36 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | reject / correct | [open](case_36/solo/r1/transcript.md) |
| correlation_pair | case_37 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | trust / correct | [open](case_37/debate/r1/transcript.md) |
| correlation_pair | case_37 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | None / token_limit | [open](case_37/single_call/r1/transcript.md) |
| correlation_pair | case_37 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | historical | reject / wrong_verdict | [open](case_37/single_call/r2/transcript.md) |
| correlation_pair | case_37 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | historical | trust / correct | [open](case_37/solo/r1/transcript.md) |
| numerical_challenges | case_38 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_38/debate/r1/trace_meta.json) |
| numerical_challenges | case_38 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_38/debate/r2/transcript.md) |
| numerical_challenges | case_38 | debate | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_38/debate/r3/transcript.md) |
| numerical_challenges | case_38 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_38/single_call/r1/trace_meta.json) |
| numerical_challenges | case_38 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_38/single_call/r2/trace_meta.json) |
| numerical_challenges | case_38 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | None / token_limit | [open](case_38/single_call/r3/transcript.md) |
| numerical_challenges | case_38 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | None / token_limit | [open](case_38/single_call/r4/transcript.md) |
| numerical_challenges | case_38 | single_call | r5 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_38/single_call/r5/transcript.md) |
| numerical_challenges | case_38 | single_call | r6 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_38/single_call/r6/transcript.md) |
| numerical_challenges | case_38 | single_call | r7 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_38/single_call/r7/transcript.md) |
| numerical_challenges | case_38 | single_call | r8 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_38/single_call/r8/transcript.md) |
| numerical_challenges | case_38 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_38/solo/r1/transcript.md) |
| numerical_challenges | case_38 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_38/solo/r2/transcript.md) |
| numerical_challenges | case_38 | solo | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_38/solo/r3/transcript.md) |
| numerical_challenges | case_39 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_39/debate/r1/trace_meta.json) |
| numerical_challenges | case_39 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_39/debate/r2/transcript.md) |
| numerical_challenges | case_39 | debate | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_39/debate/r3/transcript.md) |
| numerical_challenges | case_39 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_39/single_call/r1/transcript.md) |
| numerical_challenges | case_39 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_39/single_call/r2/trace_meta.json) |
| numerical_challenges | case_39 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | None / token_limit | [open](case_39/single_call/r3/transcript.md) |
| numerical_challenges | case_39 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_39/single_call/r4/transcript.md) |
| numerical_challenges | case_39 | single_call | r5 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_39/single_call/r5/transcript.md) |
| numerical_challenges | case_39 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_39/solo/r1/trace_meta.json) |
| numerical_challenges | case_39 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_39/solo/r2/transcript.md) |
| numerical_challenges | case_39 | solo | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_39/solo/r3/transcript.md) |
| numerical_challenges | case_40 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_40/debate/r1/trace_meta.json) |
| numerical_challenges | case_40 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_40/debate/r2/transcript.md) |
| numerical_challenges | case_40 | debate | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_40/debate/r3/transcript.md) |
| numerical_challenges | case_40 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_40/single_call/r1/trace_meta.json) |
| numerical_challenges | case_40 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_40/single_call/r2/trace_meta.json) |
| numerical_challenges | case_40 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_40/single_call/r3/transcript.md) |
| numerical_challenges | case_40 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | None / token_limit | [open](case_40/single_call/r4/transcript.md) |
| numerical_challenges | case_40 | single_call | r5 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_40/single_call/r5/transcript.md) |
| numerical_challenges | case_40 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_40/solo/r1/trace_meta.json) |
| numerical_challenges | case_40 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_40/solo/r2/transcript.md) |
| numerical_challenges | case_40 | solo | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_40/solo/r3/transcript.md) |
| numerical_challenges | case_41 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_41/debate/r1/trace_meta.json) |
| numerical_challenges | case_41 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_41/debate/r2/transcript.md) |
| numerical_challenges | case_41 | debate | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_41/debate/r3/transcript.md) |
| numerical_challenges | case_41 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_41/single_call/r1/trace_meta.json) |
| numerical_challenges | case_41 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_41/single_call/r2/transcript.md) |
| numerical_challenges | case_41 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_41/single_call/r3/transcript.md) |
| numerical_challenges | case_41 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_41/single_call/r4/transcript.md) |
| numerical_challenges | case_41 | single_call | r5 | accounts/fireworks/models/glm-5p3 / fireworks | completed | None / token_limit | [open](case_41/single_call/r5/transcript.md) |
| numerical_challenges | case_41 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_41/solo/r1/trace_meta.json) |
| numerical_challenges | case_41 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_41/solo/r2/transcript.md) |
| numerical_challenges | case_41 | solo | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_41/solo/r3/transcript.md) |
| numerical_challenges | case_42 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_42/debate/r1/trace_meta.json) |
| numerical_challenges | case_42 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/debate/r2/transcript.md) |
| numerical_challenges | case_42 | debate | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/debate/r3/transcript.md) |
| numerical_challenges | case_42 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_42/single_call/r1/trace_meta.json) |
| numerical_challenges | case_42 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/single_call/r2/transcript.md) |
| numerical_challenges | case_42 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/single_call/r3/transcript.md) |
| numerical_challenges | case_42 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/single_call/r4/transcript.md) |
| numerical_challenges | case_42 | single_call | r5 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/single_call/r5/transcript.md) |
| numerical_challenges | case_42 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_42/solo/r1/trace_meta.json) |
| numerical_challenges | case_42 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/solo/r2/transcript.md) |
| numerical_challenges | case_42 | solo | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_42/solo/r3/transcript.md) |
| numerical_challenges | case_43 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_43/debate/r1/trace_meta.json) |
| numerical_challenges | case_43 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_43/debate/r2/transcript.md) |
| numerical_challenges | case_43 | debate | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_43/debate/r3/transcript.md) |
| numerical_challenges | case_43 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | running | None / no_verdict | [open](case_43/single_call/r1/trace_meta.json) |
| numerical_challenges | case_43 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_43/single_call/r2/transcript.md) |
| numerical_challenges | case_43 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_43/single_call/r3/transcript.md) |
| numerical_challenges | case_43 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_43/single_call/r4/transcript.md) |
| numerical_challenges | case_43 | single_call | r5 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_43/single_call/r5/transcript.md) |
| numerical_challenges | case_43 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_43/solo/r1/trace_meta.json) |
| numerical_challenges | case_43 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_43/solo/r2/transcript.md) |
| numerical_challenges | case_43 | solo | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_43/solo/r3/transcript.md) |
| numerical_challenges | case_44 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_44/debate/r1/transcript.md) |
| numerical_challenges | case_44 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_44/debate/r2/transcript.md) |
| numerical_challenges | case_44 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_44/single_call/r1/transcript.md) |
| numerical_challenges | case_44 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_44/single_call/r2/transcript.md) |
| numerical_challenges | case_44 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_44/single_call/r3/transcript.md) |
| numerical_challenges | case_44 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_44/single_call/r4/transcript.md) |
| numerical_challenges | case_44 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_44/solo/r1/transcript.md) |
| numerical_challenges | case_44 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_44/solo/r2/transcript.md) |
| numerical_challenges | case_45 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/debate/r1/transcript.md) |
| numerical_challenges | case_45 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/debate/r2/transcript.md) |
| numerical_challenges | case_45 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/single_call/r1/transcript.md) |
| numerical_challenges | case_45 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/single_call/r2/transcript.md) |
| numerical_challenges | case_45 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/single_call/r3/transcript.md) |
| numerical_challenges | case_45 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/single_call/r4/transcript.md) |
| numerical_challenges | case_45 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/solo/r1/transcript.md) |
| numerical_challenges | case_45 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_45/solo/r2/transcript.md) |
| numerical_challenges | case_46 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_46/debate/r1/transcript.md) |
| numerical_challenges | case_46 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_46/debate/r2/transcript.md) |
| numerical_challenges | case_46 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_46/single_call/r1/transcript.md) |
| numerical_challenges | case_46 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_46/single_call/r2/transcript.md) |
| numerical_challenges | case_46 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_46/single_call/r3/transcript.md) |
| numerical_challenges | case_46 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_46/single_call/r4/transcript.md) |
| numerical_challenges | case_46 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_46/solo/r1/transcript.md) |
| numerical_challenges | case_46 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_46/solo/r2/transcript.md) |
| numerical_challenges | case_47 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/debate/r1/transcript.md) |
| numerical_challenges | case_47 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/debate/r2/transcript.md) |
| numerical_challenges | case_47 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/single_call/r1/transcript.md) |
| numerical_challenges | case_47 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/single_call/r2/transcript.md) |
| numerical_challenges | case_47 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/single_call/r3/transcript.md) |
| numerical_challenges | case_47 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/single_call/r4/transcript.md) |
| numerical_challenges | case_47 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/solo/r1/transcript.md) |
| numerical_challenges | case_47 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_47/solo/r2/transcript.md) |
| numerical_challenges | case_48 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_48/debate/r1/transcript.md) |
| numerical_challenges | case_48 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_48/debate/r2/transcript.md) |
| numerical_challenges | case_48 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_48/single_call/r1/transcript.md) |
| numerical_challenges | case_48 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_48/single_call/r2/trace_meta.json) |
| numerical_challenges | case_48 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_48/single_call/r3/transcript.md) |
| numerical_challenges | case_48 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_48/single_call/r4/transcript.md) |
| numerical_challenges | case_48 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_48/solo/r1/transcript.md) |
| numerical_challenges | case_48 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_48/solo/r2/transcript.md) |
| numerical_challenges | case_49 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_49/debate/r1/transcript.md) |
| numerical_challenges | case_49 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_49/debate/r2/transcript.md) |
| numerical_challenges | case_49 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_49/single_call/r1/transcript.md) |
| numerical_challenges | case_49 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_49/single_call/r2/trace_meta.json) |
| numerical_challenges | case_49 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_49/single_call/r3/transcript.md) |
| numerical_challenges | case_49 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_49/single_call/r4/transcript.md) |
| numerical_challenges | case_49 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_49/solo/r1/transcript.md) |
| numerical_challenges | case_49 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_49/solo/r2/transcript.md) |
| numerical_challenges | case_50 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_50/debate/r1/transcript.md) |
| numerical_challenges | case_50 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_50/debate/r2/transcript.md) |
| numerical_challenges | case_50 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_50/single_call/r1/transcript.md) |
| numerical_challenges | case_50 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_50/single_call/r2/trace_meta.json) |
| numerical_challenges | case_50 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_50/single_call/r3/transcript.md) |
| numerical_challenges | case_50 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_50/single_call/r4/transcript.md) |
| numerical_challenges | case_50 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_50/solo/r1/transcript.md) |
| numerical_challenges | case_50 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_50/solo/r2/transcript.md) |
| numerical_challenges | case_51 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_51/debate/r1/transcript.md) |
| numerical_challenges | case_51 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_51/debate/r2/transcript.md) |
| numerical_challenges | case_51 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_51/single_call/r1/transcript.md) |
| numerical_challenges | case_51 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_51/single_call/r2/transcript.md) |
| numerical_challenges | case_51 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_51/single_call/r3/transcript.md) |
| numerical_challenges | case_51 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_51/single_call/r4/transcript.md) |
| numerical_challenges | case_51 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_51/solo/r1/transcript.md) |
| numerical_challenges | case_51 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_51/solo/r2/transcript.md) |
| numerical_challenges | case_52 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_52/debate/r1/transcript.md) |
| numerical_challenges | case_52 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_52/debate/r2/transcript.md) |
| numerical_challenges | case_52 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_52/single_call/r1/transcript.md) |
| numerical_challenges | case_52 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_52/single_call/r2/transcript.md) |
| numerical_challenges | case_52 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_52/single_call/r3/transcript.md) |
| numerical_challenges | case_52 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_52/single_call/r4/transcript.md) |
| numerical_challenges | case_52 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_52/solo/r1/transcript.md) |
| numerical_challenges | case_52 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_52/solo/r2/transcript.md) |
| numerical_challenges | case_53 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/debate/r1/transcript.md) |
| numerical_challenges | case_53 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/debate/r2/transcript.md) |
| numerical_challenges | case_53 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/single_call/r1/transcript.md) |
| numerical_challenges | case_53 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/single_call/r2/transcript.md) |
| numerical_challenges | case_53 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/single_call/r3/transcript.md) |
| numerical_challenges | case_53 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/single_call/r4/transcript.md) |
| numerical_challenges | case_53 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/solo/r1/transcript.md) |
| numerical_challenges | case_53 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_53/solo/r2/transcript.md) |
| numerical_challenges | case_54 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_54/debate/r1/transcript.md) |
| numerical_challenges | case_54 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_54/debate/r2/transcript.md) |
| numerical_challenges | case_54 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_54/single_call/r1/trace_meta.json) |
| numerical_challenges | case_54 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_54/single_call/r2/transcript.md) |
| numerical_challenges | case_54 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_54/single_call/r3/transcript.md) |
| numerical_challenges | case_54 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_54/single_call/r4/transcript.md) |
| numerical_challenges | case_54 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_54/solo/r1/transcript.md) |
| numerical_challenges | case_54 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_54/solo/r2/transcript.md) |
| numerical_challenges | case_55 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_55/debate/r1/transcript.md) |
| numerical_challenges | case_55 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_55/debate/r2/transcript.md) |
| numerical_challenges | case_55 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_55/single_call/r1/trace_meta.json) |
| numerical_challenges | case_55 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_55/single_call/r2/transcript.md) |
| numerical_challenges | case_55 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_55/single_call/r3/transcript.md) |
| numerical_challenges | case_55 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_55/single_call/r4/transcript.md) |
| numerical_challenges | case_55 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_55/solo/r1/transcript.md) |
| numerical_challenges | case_55 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_55/solo/r2/transcript.md) |
| numerical_challenges | case_56 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_56/debate/r1/transcript.md) |
| numerical_challenges | case_56 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_56/debate/r2/transcript.md) |
| numerical_challenges | case_56 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_56/single_call/r1/transcript.md) |
| numerical_challenges | case_56 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_56/single_call/r2/transcript.md) |
| numerical_challenges | case_56 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_56/single_call/r3/transcript.md) |
| numerical_challenges | case_56 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_56/single_call/r4/transcript.md) |
| numerical_challenges | case_56 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_56/solo/r1/transcript.md) |
| numerical_challenges | case_56 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_56/solo/r2/transcript.md) |
| numerical_challenges | case_57 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_57/debate/r1/transcript.md) |
| numerical_challenges | case_57 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_57/debate/r2/transcript.md) |
| numerical_challenges | case_57 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_57/single_call/r1/transcript.md) |
| numerical_challenges | case_57 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_57/single_call/r2/trace_meta.json) |
| numerical_challenges | case_57 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_57/single_call/r3/transcript.md) |
| numerical_challenges | case_57 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_57/single_call/r4/transcript.md) |
| numerical_challenges | case_57 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_57/solo/r1/transcript.md) |
| numerical_challenges | case_57 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_57/solo/r2/transcript.md) |
| numerical_challenges | case_58 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/debate/r1/transcript.md) |
| numerical_challenges | case_58 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/debate/r2/transcript.md) |
| numerical_challenges | case_58 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/single_call/r1/transcript.md) |
| numerical_challenges | case_58 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/single_call/r2/transcript.md) |
| numerical_challenges | case_58 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/single_call/r3/transcript.md) |
| numerical_challenges | case_58 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_58/single_call/r4/trace_meta.json) |
| numerical_challenges | case_58 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/solo/r1/transcript.md) |
| numerical_challenges | case_58 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_58/solo/r2/transcript.md) |
| numerical_challenges | case_59 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_59/debate/r1/transcript.md) |
| numerical_challenges | case_59 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_59/debate/r2/transcript.md) |
| numerical_challenges | case_59 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_59/single_call/r1/transcript.md) |
| numerical_challenges | case_59 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_59/single_call/r2/transcript.md) |
| numerical_challenges | case_59 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_59/single_call/r3/transcript.md) |
| numerical_challenges | case_59 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_59/single_call/r4/trace_meta.json) |
| numerical_challenges | case_59 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_59/solo/r1/transcript.md) |
| numerical_challenges | case_59 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_59/solo/r2/transcript.md) |
| numerical_challenges | case_60 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/debate/r1/transcript.md) |
| numerical_challenges | case_60 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/debate/r2/transcript.md) |
| numerical_challenges | case_60 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_60/single_call/r1/transcript.md) |
| numerical_challenges | case_60 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/single_call/r2/transcript.md) |
| numerical_challenges | case_60 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/single_call/r3/transcript.md) |
| numerical_challenges | case_60 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/single_call/r4/transcript.md) |
| numerical_challenges | case_60 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/solo/r1/transcript.md) |
| numerical_challenges | case_60 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_60/solo/r2/transcript.md) |
| numerical_challenges | case_61 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_61/debate/r1/transcript.md) |
| numerical_challenges | case_61 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_61/debate/r2/transcript.md) |
| numerical_challenges | case_61 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_61/single_call/r1/transcript.md) |
| numerical_challenges | case_61 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_61/single_call/r2/transcript.md) |
| numerical_challenges | case_61 | single_call | r3 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_61/single_call/r3/transcript.md) |
| numerical_challenges | case_61 | single_call | r4 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_61/single_call/r4/transcript.md) |
| numerical_challenges | case_61 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_61/solo/r1/transcript.md) |
| numerical_challenges | case_61 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_61/solo/r2/transcript.md) |

### Solo vs 多角色 debate

| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| evidence_challenges | case_62 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_62/debate/r1/transcript.md) |
| evidence_challenges | case_62 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_62/single_call/r1/transcript.md) |
| evidence_challenges | case_62 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_62/solo/r1/transcript.md) |
| evidence_challenges | case_63 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_63/debate/r1/transcript.md) |
| evidence_challenges | case_63 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_63/single_call/r1/transcript.md) |
| evidence_challenges | case_63 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_63/solo/r1/transcript.md) |
| evidence_challenges | case_64 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_64/debate/r1/transcript.md) |
| evidence_challenges | case_64 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_64/single_call/r1/transcript.md) |
| evidence_challenges | case_64 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_64/solo/r1/transcript.md) |
| evidence_challenges | case_65 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_65/debate/r1/transcript.md) |
| evidence_challenges | case_65 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_65/single_call/r1/transcript.md) |
| evidence_challenges | case_65 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_65/solo/r1/transcript.md) |
| evidence_challenges | case_66 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_66/debate/r1/transcript.md) |
| evidence_challenges | case_66 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_66/single_call/r1/transcript.md) |
| evidence_challenges | case_66 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_66/solo/r1/transcript.md) |
| evidence_challenges | case_67 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_67/debate/r1/transcript.md) |
| evidence_challenges | case_67 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_67/single_call/r1/transcript.md) |
| evidence_challenges | case_67 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_67/solo/r1/transcript.md) |
| evidence_challenges | case_68 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_68/debate/r1/transcript.md) |
| evidence_challenges | case_68 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_68/single_call/r1/transcript.md) |
| evidence_challenges | case_68 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_68/solo/r1/transcript.md) |
| evidence_challenges | case_69 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_69/debate/r1/transcript.md) |
| evidence_challenges | case_69 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_69/single_call/r1/transcript.md) |
| evidence_challenges | case_69 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_69/solo/r1/transcript.md) |
| evidence_challenges | case_70 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_70/debate/r1/transcript.md) |
| evidence_challenges | case_70 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_70/single_call/r1/transcript.md) |
| evidence_challenges | case_70 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_70/solo/r1/transcript.md) |
| evidence_challenges | case_71 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_71/debate/r1/transcript.md) |
| evidence_challenges | case_71 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_71/single_call/r1/transcript.md) |
| evidence_challenges | case_71 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_71/solo/r1/transcript.md) |
| evidence_challenges | case_72 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_72/debate/r1/transcript.md) |
| evidence_challenges | case_72 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_72/debate/r2/transcript.md) |
| evidence_challenges | case_72 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_72/single_call/r1/transcript.md) |
| evidence_challenges | case_72 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_72/single_call/r2/transcript.md) |
| evidence_challenges | case_72 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_72/solo/r1/transcript.md) |
| evidence_challenges | case_72 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_72/solo/r2/transcript.md) |
| evidence_challenges | case_73 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_73/debate/r1/transcript.md) |
| evidence_challenges | case_73 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_73/debate/r2/transcript.md) |
| evidence_challenges | case_73 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_73/single_call/r1/transcript.md) |
| evidence_challenges | case_73 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_73/single_call/r2/transcript.md) |
| evidence_challenges | case_73 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_73/solo/r1/transcript.md) |
| evidence_challenges | case_73 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_73/solo/r2/transcript.md) |
| evidence_challenges | case_74 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_74/debate/r1/transcript.md) |
| evidence_challenges | case_74 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_74/debate/r2/transcript.md) |
| evidence_challenges | case_74 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_74/single_call/r1/transcript.md) |
| evidence_challenges | case_74 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_74/single_call/r2/transcript.md) |
| evidence_challenges | case_74 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_74/solo/r1/transcript.md) |
| evidence_challenges | case_74 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_74/solo/r2/transcript.md) |
| evidence_challenges | case_75 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_75/debate/r1/transcript.md) |
| evidence_challenges | case_75 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_75/debate/r2/transcript.md) |
| evidence_challenges | case_75 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_75/single_call/r1/transcript.md) |
| evidence_challenges | case_75 | single_call | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_75/single_call/r2/transcript.md) |
| evidence_challenges | case_75 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_75/solo/r1/transcript.md) |
| evidence_challenges | case_75 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_75/solo/r2/transcript.md) |
| evidence_challenges | case_76 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_76/debate/r1/transcript.md) |
| evidence_challenges | case_76 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_76/single_call/r1/transcript.md) |
| evidence_challenges | case_76 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_76/solo/r1/transcript.md) |
| evidence_challenges | case_77 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_77/debate/r1/transcript.md) |
| evidence_challenges | case_77 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_77/single_call/r1/transcript.md) |
| evidence_challenges | case_77 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_77/solo/r1/transcript.md) |
| evidence_challenges | case_78 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_78/debate/r1/transcript.md) |
| evidence_challenges | case_78 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_78/single_call/r1/transcript.md) |
| evidence_challenges | case_78 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_78/solo/r1/transcript.md) |
| evidence_challenges | case_79 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_79/debate/r1/transcript.md) |
| evidence_challenges | case_79 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_79/single_call/r1/transcript.md) |
| evidence_challenges | case_79 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_79/solo/r1/transcript.md) |
| evidence_challenges | case_80 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_80/debate/r1/transcript.md) |
| evidence_challenges | case_80 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_80/single_call/r1/transcript.md) |
| evidence_challenges | case_80 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_80/solo/r1/transcript.md) |
| evidence_challenges | case_81 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_81/debate/r1/transcript.md) |
| evidence_challenges | case_81 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_81/single_call/r1/transcript.md) |
| evidence_challenges | case_81 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_81/solo/r1/transcript.md) |

### 专门构造的真实 kernel 验证挑战（单独统计）

| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_kernel_challenges | case_106 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_106/debate/r1/transcript.md) |
| real_kernel_challenges | case_106 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_106/single_call/r1/transcript.md) |
| real_kernel_challenges | case_106 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_106/solo/r1/transcript.md) |
| real_kernel_challenges | case_107 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_107/debate/r1/transcript.md) |
| real_kernel_challenges | case_107 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_107/single_call/r1/transcript.md) |
| real_kernel_challenges | case_107 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_107/solo/r1/transcript.md) |
| real_kernel_challenges | case_108 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_108/debate/r1/trace_meta.json) |
| real_kernel_challenges | case_108 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_108/debate/r2/transcript.md) |
| real_kernel_challenges | case_108 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_108/single_call/r1/transcript.md) |
| real_kernel_challenges | case_108 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_108/solo/r1/trace_meta.json) |
| real_kernel_challenges | case_108 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_108/solo/r2/transcript.md) |
| real_kernel_challenges | case_109 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_109/debate/r1/trace_meta.json) |
| real_kernel_challenges | case_109 | debate | r2 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / token_limit | [open](case_109/debate/r2/transcript.md) |
| real_kernel_challenges | case_109 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_109/single_call/r1/transcript.md) |
| real_kernel_challenges | case_109 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | error | None / no_verdict | [open](case_109/solo/r1/trace_meta.json) |
| real_kernel_challenges | case_109 | solo | r2 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_109/solo/r2/transcript.md) |
| real_kernel_challenges | case_110 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_110/debate/r1/transcript.md) |
| real_kernel_challenges | case_110 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_110/single_call/r1/transcript.md) |
| real_kernel_challenges | case_110 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_110/solo/r1/transcript.md) |
| real_kernel_challenges | case_111 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_111/debate/r1/transcript.md) |
| real_kernel_challenges | case_111 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_111/single_call/r1/transcript.md) |
| real_kernel_challenges | case_111 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_111/solo/r1/transcript.md) |
| real_kernel_challenges | case_112 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_112/debate/r1/transcript.md) |
| real_kernel_challenges | case_112 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_112/single_call/r1/transcript.md) |
| real_kernel_challenges | case_112 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_112/solo/r1/transcript.md) |
| real_kernel_challenges | case_113 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_113/debate/r1/transcript.md) |
| real_kernel_challenges | case_113 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_113/single_call/r1/transcript.md) |
| real_kernel_challenges | case_113 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_113/solo/r1/transcript.md) |
| real_kernel_challenges | case_114 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_114/debate/r1/transcript.md) |
| real_kernel_challenges | case_114 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_114/single_call/r1/transcript.md) |
| real_kernel_challenges | case_114 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_114/solo/r1/transcript.md) |
| real_kernel_challenges | case_115 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_115/debate/r1/transcript.md) |
| real_kernel_challenges | case_115 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_115/single_call/r1/transcript.md) |
| real_kernel_challenges | case_115 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_115/solo/r1/transcript.md) |

430 recorded GLM trials. See [trace format](../TRACES.md) for provenance and capture limits.
