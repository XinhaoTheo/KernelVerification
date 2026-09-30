# GLM trace index

All GLM runs are listed together by case, arm and trial. API provenance is retained in trace metadata. Historical tool traces do not have raw API payloads; new runs include `llm_calls/`.

See [案例总索引](../CASE_INDEX.md) for numeric ranges, original IDs and source kernels. Historical trace payloads keep their original IDs; the current directory and registry determine the case.

每个 case/arm 下按历史时间顺序编号为 `r1`、`r2`……；不同案例的同名 rN 不代表同一实验批次。原始批次保存在 `trace_meta.json` 的 `original_trial`，统计继续按批次、模型和预算配置分组。

## 全题库三组覆盖

范围：case_map.json 中全部 104 个活跃案例，共 312 个 case-arm 槽位。
完成包含正确、错误和 needs_more_evidence（弃答）；截断、无最终判断、运行错误和运行中均不算完成。错误答案和有效弃答保留，不因结果不理想而补跑。

| 实验组 | 已完成 | 从未运行 | 失败/未完成 | 运行中 | 未完成合计 |
|---|---:|---:|---:|---:|---:|
| single_call | 104 | 0 | 0 | 0 | 0 |
| solo | 104 | 0 | 0 | 0 | 0 |
| debate | 104 | 0 | 0 | 0 | 0 |

完成记录按最早的 created_at 选取；缺少日期的旧记录优先，并以迁移前路径稳定排序。不按正确性或置信度筛选；下方历史索引仍保留全部尝试。
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
| case_36 | correlation_pair | [r1](case_36/single_call/r1/transcript.md) · reject / 正确 | [r1](case_36/solo/r1/transcript.md) · reject / 正确 | [r1](case_36/debate/r1/transcript.md) · reject / 正确 |
| case_37 | correlation_pair | [r2](case_37/single_call/r2/transcript.md) · reject / 错误 | [r1](case_37/solo/r1/transcript.md) · trust / 正确 | [r1](case_37/debate/r1/transcript.md) · trust / 正确 |
| case_38 | numerical_challenges | [r5](case_38/single_call/r5/transcript.md) · reject / 错误 | [r1](case_38/solo/r1/transcript.md) · trust / 正确 | [r2](case_38/debate/r2/transcript.md) · trust / 正确 |
| case_39 | numerical_challenges | [r1](case_39/single_call/r1/transcript.md) · reject / 正确 | [r2](case_39/solo/r2/transcript.md) · reject / 正确 | [r2](case_39/debate/r2/transcript.md) · reject / 正确 |
| case_40 | numerical_challenges | [r3](case_40/single_call/r3/transcript.md) · trust / 正确 | [r2](case_40/solo/r2/transcript.md) · trust / 正确 | [r2](case_40/debate/r2/transcript.md) · trust / 正确 |
| case_41 | numerical_challenges | [r2](case_41/single_call/r2/transcript.md) · trust / 错误 | [r2](case_41/solo/r2/transcript.md) · reject / 正确 | [r2](case_41/debate/r2/transcript.md) · reject / 正确 |
| case_42 | numerical_challenges | [r2](case_42/single_call/r2/transcript.md) · reject / 正确 | [r2](case_42/solo/r2/transcript.md) · reject / 正确 | [r2](case_42/debate/r2/transcript.md) · reject / 正确 |
| case_43 | numerical_challenges | [r2](case_43/single_call/r2/transcript.md) · reject / 错误 | [r2](case_43/solo/r2/transcript.md) · trust / 正确 | [r2](case_43/debate/r2/transcript.md) · trust / 正确 |
| case_44 | numerical_challenges | [r1](case_44/single_call/r1/transcript.md) · reject / 错误 | [r1](case_44/solo/r1/transcript.md) · trust / 正确 | [r1](case_44/debate/r1/transcript.md) · trust / 正确 |
| case_45 | numerical_challenges | [r1](case_45/single_call/r1/transcript.md) · reject / 正确 | [r1](case_45/solo/r1/transcript.md) · reject / 正确 | [r1](case_45/debate/r1/transcript.md) · reject / 正确 |
| case_46 | numerical_challenges | [r1](case_46/single_call/r1/transcript.md) · reject / 错误 | [r1](case_46/solo/r1/transcript.md) · trust / 正确 | [r1](case_46/debate/r1/transcript.md) · trust / 正确 |
| case_47 | numerical_challenges | [r1](case_47/single_call/r1/transcript.md) · reject / 正确 | [r1](case_47/solo/r1/transcript.md) · reject / 正确 | [r1](case_47/debate/r1/transcript.md) · reject / 正确 |
| case_48 | numerical_challenges | [r1](case_48/single_call/r1/transcript.md) · reject / 错误 | [r1](case_48/solo/r1/transcript.md) · trust / 正确 | [r1](case_48/debate/r1/transcript.md) · trust / 正确 |
| case_49 | numerical_challenges | [r1](case_49/single_call/r1/transcript.md) · reject / 正确 | [r1](case_49/solo/r1/transcript.md) · reject / 正确 | [r1](case_49/debate/r1/transcript.md) · reject / 正确 |
| case_50 | numerical_challenges | [r1](case_50/single_call/r1/transcript.md) · reject / 错误 | [r1](case_50/solo/r1/transcript.md) · trust / 正确 | [r1](case_50/debate/r1/transcript.md) · trust / 正确 |
| case_51 | numerical_challenges | [r1](case_51/single_call/r1/transcript.md) · reject / 正确 | [r1](case_51/solo/r1/transcript.md) · reject / 正确 | [r1](case_51/debate/r1/transcript.md) · reject / 正确 |
| case_52 | numerical_challenges | [r1](case_52/single_call/r1/transcript.md) · reject / 错误 | [r1](case_52/solo/r1/transcript.md) · trust / 正确 | [r1](case_52/debate/r1/transcript.md) · trust / 正确 |
| case_53 | numerical_challenges | [r1](case_53/single_call/r1/transcript.md) · reject / 正确 | [r1](case_53/solo/r1/transcript.md) · reject / 正确 | [r1](case_53/debate/r1/transcript.md) · reject / 正确 |
| case_54 | numerical_challenges | [r2](case_54/single_call/r2/transcript.md) · reject / 错误 | [r1](case_54/solo/r1/transcript.md) · trust / 正确 | [r1](case_54/debate/r1/transcript.md) · trust / 正确 |
| case_55 | numerical_challenges | [r2](case_55/single_call/r2/transcript.md) · trust / 错误 | [r1](case_55/solo/r1/transcript.md) · reject / 正确 | [r1](case_55/debate/r1/transcript.md) · reject / 正确 |
| case_56 | numerical_challenges | [r1](case_56/single_call/r1/transcript.md) · reject / 错误 | [r1](case_56/solo/r1/transcript.md) · trust / 正确 | [r1](case_56/debate/r1/transcript.md) · trust / 正确 |
| case_57 | numerical_challenges | [r1](case_57/single_call/r1/transcript.md) · needs_more_evidence / 弃答 | [r1](case_57/solo/r1/transcript.md) · reject / 正确 | [r1](case_57/debate/r1/transcript.md) · reject / 正确 |
| case_58 | numerical_challenges | [r1](case_58/single_call/r1/transcript.md) · trust / 正确 | [r1](case_58/solo/r1/transcript.md) · trust / 正确 | [r1](case_58/debate/r1/transcript.md) · trust / 正确 |
| case_59 | numerical_challenges | [r1](case_59/single_call/r1/transcript.md) · trust / 错误 | [r1](case_59/solo/r1/transcript.md) · reject / 正确 | [r1](case_59/debate/r1/transcript.md) · reject / 正确 |
| case_60 | numerical_challenges | [r1](case_60/single_call/r1/transcript.md) · reject / 错误 | [r1](case_60/solo/r1/transcript.md) · trust / 正确 | [r1](case_60/debate/r1/transcript.md) · trust / 正确 |
| case_61 | numerical_challenges | [r1](case_61/single_call/r1/transcript.md) · trust / 错误 | [r1](case_61/solo/r1/transcript.md) · reject / 正确 | [r1](case_61/debate/r1/transcript.md) · reject / 正确 |
| case_62 | evidence_challenges | [r1](case_62/single_call/r1/transcript.md) · trust / 正确 | [r1](case_62/solo/r1/transcript.md) · trust / 正确 | [r1](case_62/debate/r1/transcript.md) · trust / 正确 |
| case_63 | evidence_challenges | [r1](case_63/single_call/r1/transcript.md) · reject / 正确 | [r1](case_63/solo/r1/transcript.md) · reject / 正确 | [r1](case_63/debate/r1/transcript.md) · reject / 正确 |
| case_64 | evidence_challenges | [r1](case_64/single_call/r1/transcript.md) · reject / 错误 | [r1](case_64/solo/r1/transcript.md) · trust / 正确 | [r1](case_64/debate/r1/transcript.md) · trust / 正确 |
| case_65 | evidence_challenges | [r1](case_65/single_call/r1/transcript.md) · reject / 正确 | [r1](case_65/solo/r1/transcript.md) · reject / 正确 | [r1](case_65/debate/r1/transcript.md) · reject / 正确 |
| case_66 | evidence_challenges | [r1](case_66/single_call/r1/transcript.md) · needs_more_evidence / 弃答 | [r1](case_66/solo/r1/transcript.md) · trust / 正确 | [r1](case_66/debate/r1/transcript.md) · trust / 正确 |
| case_67 | evidence_challenges | [r1](case_67/single_call/r1/transcript.md) · trust / 错误 | [r1](case_67/solo/r1/transcript.md) · reject / 正确 | [r1](case_67/debate/r1/transcript.md) · reject / 正确 |
| case_68 | evidence_challenges | [r1](case_68/single_call/r1/transcript.md) · trust / 正确 | [r1](case_68/solo/r1/transcript.md) · trust / 正确 | [r1](case_68/debate/r1/transcript.md) · trust / 正确 |
| case_69 | evidence_challenges | [r1](case_69/single_call/r1/transcript.md) · reject / 正确 | [r1](case_69/solo/r1/transcript.md) · reject / 正确 | [r1](case_69/debate/r1/transcript.md) · reject / 正确 |
| case_70 | evidence_challenges | [r1](case_70/single_call/r1/transcript.md) · reject / 错误 | [r1](case_70/solo/r1/transcript.md) · trust / 正确 | [r1](case_70/debate/r1/transcript.md) · trust / 正确 |
| case_71 | evidence_challenges | [r1](case_71/single_call/r1/transcript.md) · reject / 正确 | [r1](case_71/solo/r1/transcript.md) · reject / 正确 | [r1](case_71/debate/r1/transcript.md) · reject / 正确 |
| case_72 | evidence_challenges | [r1](case_72/single_call/r1/transcript.md) · reject / 错误 | [r1](case_72/solo/r1/transcript.md) · trust / 正确 | [r1](case_72/debate/r1/transcript.md) · trust / 正确 |
| case_73 | evidence_challenges | [r1](case_73/single_call/r1/transcript.md) · reject / 正确 | [r1](case_73/solo/r1/transcript.md) · reject / 正确 | [r1](case_73/debate/r1/transcript.md) · reject / 正确 |
| case_74 | evidence_challenges | [r1](case_74/single_call/r1/transcript.md) · reject / 错误 | [r1](case_74/solo/r1/transcript.md) · reject / 错误 | [r1](case_74/debate/r1/transcript.md) · trust / 正确 |
| case_75 | evidence_challenges | [r1](case_75/single_call/r1/transcript.md) · trust / 错误 | [r1](case_75/solo/r1/transcript.md) · reject / 正确 | [r1](case_75/debate/r1/transcript.md) · needs_more_evidence / 弃答 |
| case_76 | evidence_challenges | [r1](case_76/single_call/r1/transcript.md) · reject / 错误 | [r1](case_76/solo/r1/transcript.md) · trust / 正确 | [r1](case_76/debate/r1/transcript.md) · trust / 正确 |
| case_77 | evidence_challenges | [r1](case_77/single_call/r1/transcript.md) · trust / 错误 | [r1](case_77/solo/r1/transcript.md) · reject / 正确 | [r1](case_77/debate/r1/transcript.md) · reject / 正确 |
| case_78 | evidence_challenges | [r1](case_78/single_call/r1/transcript.md) · trust / 正确 | [r1](case_78/solo/r1/transcript.md) · reject / 错误 | [r1](case_78/debate/r1/transcript.md) · trust / 正确 |
| case_79 | evidence_challenges | [r1](case_79/single_call/r1/transcript.md) · trust / 错误 | [r1](case_79/solo/r1/transcript.md) · reject / 正确 | [r1](case_79/debate/r1/transcript.md) · reject / 正确 |
| case_80 | evidence_challenges | [r1](case_80/single_call/r1/transcript.md) · trust / 正确 | [r1](case_80/solo/r1/transcript.md) · trust / 正确 | [r1](case_80/debate/r1/transcript.md) · trust / 正确 |
| case_81 | evidence_challenges | [r1](case_81/single_call/r1/transcript.md) · trust / 错误 | [r1](case_81/solo/r1/transcript.md) · reject / 正确 | [r1](case_81/debate/r1/transcript.md) · reject / 正确 |
| case_82 | numerical_pilot | [r1](case_82/single_call/r1/transcript.md) · reject / 正确 | [r1](case_82/solo/r1/transcript.md) · reject / 正确 | [r1](case_82/debate/r1/transcript.md) · reject / 正确 |
| case_83 | numerical_pilot | [r1](case_83/single_call/r1/transcript.md) · reject / 正确 | [r1](case_83/solo/r1/transcript.md) · reject / 正确 | [r1](case_83/debate/r1/transcript.md) · reject / 正确 |
| case_84 | numerical_pilot | [r1](case_84/single_call/r1/transcript.md) · trust / 正确 | [r1](case_84/solo/r1/transcript.md) · trust / 正确 | [r1](case_84/debate/r1/transcript.md) · trust / 正确 |
| case_85 | numerical_pilot | [r1](case_85/single_call/r1/transcript.md) · reject / 正确 | [r1](case_85/solo/r1/transcript.md) · reject / 正确 | [r1](case_85/debate/r1/transcript.md) · reject / 正确 |
| case_86 | numerical_pilot | [r1](case_86/single_call/r1/transcript.md) · trust / 正确 | [r1](case_86/solo/r1/transcript.md) · trust / 正确 | [r1](case_86/debate/r1/transcript.md) · trust / 正确 |
| case_87 | numerical_pilot | [r1](case_87/single_call/r1/transcript.md) · trust / 错误 | [r1](case_87/solo/r1/transcript.md) · reject / 正确 | [r1](case_87/debate/r1/transcript.md) · needs_more_evidence / 弃答 |
| case_88 | numerical_pilot | [r1](case_88/single_call/r1/transcript.md) · trust / 错误 | [r1](case_88/solo/r1/transcript.md) · reject / 正确 | [r1](case_88/debate/r1/transcript.md) · reject / 正确 |
| case_89 | numerical_pilot | [r1](case_89/single_call/r1/transcript.md) · reject / 错误 | [r1](case_89/solo/r1/transcript.md) · trust / 正确 | [r1](case_89/debate/r1/transcript.md) · trust / 正确 |
| case_90 | numerical_pilot | [r1](case_90/single_call/r1/transcript.md) · trust / 错误 | [r1](case_90/solo/r1/transcript.md) · reject / 正确 | [r1](case_90/debate/r1/transcript.md) · reject / 正确 |
| case_91 | numerical_pilot | [r1](case_91/single_call/r1/transcript.md) · reject / 正确 | [r1](case_91/solo/r1/transcript.md) · reject / 正确 | [r1](case_91/debate/r1/transcript.md) · reject / 正确 |
| case_92 | numerical_pilot | [r1](case_92/single_call/r1/transcript.md) · reject / 正确 | [r1](case_92/solo/r1/transcript.md) · reject / 正确 | [r1](case_92/debate/r1/transcript.md) · reject / 正确 |
| case_93 | numerical_pilot | [r1](case_93/single_call/r1/transcript.md) · reject / 正确 | [r1](case_93/solo/r1/transcript.md) · reject / 正确 | [r1](case_93/debate/r1/transcript.md) · reject / 正确 |
| case_94 | numerical_pilot | [r1](case_94/single_call/r1/transcript.md) · reject / 正确 | [r1](case_94/solo/r1/transcript.md) · reject / 正确 | [r1](case_94/debate/r1/transcript.md) · reject / 正确 |
| case_95 | numerical_pilot | [r1](case_95/single_call/r1/transcript.md) · reject / 正确 | [r1](case_95/solo/r1/transcript.md) · reject / 正确 | [r1](case_95/debate/r1/transcript.md) · reject / 正确 |
| case_96 | numerical_pilot | [r1](case_96/single_call/r1/transcript.md) · reject / 正确 | [r1](case_96/solo/r1/transcript.md) · reject / 正确 | [r1](case_96/debate/r1/transcript.md) · reject / 正确 |
| case_97 | numerical_pilot | [r1](case_97/single_call/r1/transcript.md) · trust / 正确 | [r1](case_97/solo/r1/transcript.md) · trust / 正确 | [r1](case_97/debate/r1/transcript.md) · trust / 正确 |
| case_98 | numerical_pilot | [r1](case_98/single_call/r1/transcript.md) · needs_more_evidence / 弃答 | [r1](case_98/solo/r1/transcript.md) · trust / 正确 | [r1](case_98/debate/r1/transcript.md) · trust / 正确 |
| case_99 | numerical_pilot | [r1](case_99/single_call/r1/transcript.md) · trust / 正确 | [r1](case_99/solo/r1/transcript.md) · trust / 正确 | [r1](case_99/debate/r1/transcript.md) · trust / 正确 |
| case_100 | numerical_pilot | [r1](case_100/single_call/r1/transcript.md) · reject / 错误 | [r1](case_100/solo/r1/transcript.md) · trust / 正确 | [r1](case_100/debate/r1/transcript.md) · trust / 正确 |
| case_101 | numerical_pilot | [r1](case_101/single_call/r1/transcript.md) · trust / 正确 | [r1](case_101/solo/r1/transcript.md) · trust / 正确 | [r1](case_101/debate/r1/transcript.md) · trust / 正确 |
| case_102 | numerical_pilot | [r1](case_102/single_call/r1/transcript.md) · trust / 正确 | [r1](case_102/solo/r1/transcript.md) · trust / 正确 | [r1](case_102/debate/r1/transcript.md) · trust / 正确 |
| case_103 | numerical_pilot | [r1](case_103/single_call/r1/transcript.md) · trust / 正确 | [r1](case_103/solo/r1/transcript.md) · trust / 正确 | [r1](case_103/debate/r1/transcript.md) · trust / 正确 |
| case_104 | numerical_pilot | [r1](case_104/single_call/r1/transcript.md) · trust / 正确 | [r1](case_104/solo/r1/transcript.md) · trust / 正确 | [r1](case_104/debate/r1/transcript.md) · trust / 正确 |
| case_105 | numerical_pilot | [r1](case_105/single_call/r1/transcript.md) · trust / 正确 | [r1](case_105/solo/r1/transcript.md) · trust / 正确 | [r1](case_105/debate/r1/transcript.md) · trust / 正确 |

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

### 无工具单次调用 vs 工具：量化误差配对

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

### 无工具单次调用 vs 工具：数值误差

| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
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

### Solo vs debate：验证覆盖与参考复核

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

### 早期 24 例数值 pilot（独立旧实验）

| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| numerical_pilot | case_82 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_82/debate/r1/transcript.md) |
| numerical_pilot | case_82 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_82/single_call/r1/transcript.md) |
| numerical_pilot | case_82 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_82/solo/r1/transcript.md) |
| numerical_pilot | case_83 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_83/debate/r1/transcript.md) |
| numerical_pilot | case_83 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_83/single_call/r1/transcript.md) |
| numerical_pilot | case_83 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_83/solo/r1/transcript.md) |
| numerical_pilot | case_84 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_84/debate/r1/transcript.md) |
| numerical_pilot | case_84 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_84/single_call/r1/transcript.md) |
| numerical_pilot | case_84 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_84/solo/r1/transcript.md) |
| numerical_pilot | case_85 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_85/debate/r1/transcript.md) |
| numerical_pilot | case_85 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_85/single_call/r1/transcript.md) |
| numerical_pilot | case_85 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_85/solo/r1/transcript.md) |
| numerical_pilot | case_86 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_86/debate/r1/transcript.md) |
| numerical_pilot | case_86 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_86/single_call/r1/transcript.md) |
| numerical_pilot | case_86 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_86/solo/r1/transcript.md) |
| numerical_pilot | case_87 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_87/debate/r1/transcript.md) |
| numerical_pilot | case_87 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_87/single_call/r1/transcript.md) |
| numerical_pilot | case_87 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_87/solo/r1/transcript.md) |
| numerical_pilot | case_88 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_88/debate/r1/transcript.md) |
| numerical_pilot | case_88 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_88/single_call/r1/transcript.md) |
| numerical_pilot | case_88 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_88/solo/r1/transcript.md) |
| numerical_pilot | case_89 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_89/debate/r1/transcript.md) |
| numerical_pilot | case_89 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_89/single_call/r1/transcript.md) |
| numerical_pilot | case_89 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_89/solo/r1/transcript.md) |
| numerical_pilot | case_90 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_90/debate/r1/transcript.md) |
| numerical_pilot | case_90 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / wrong_verdict | [open](case_90/single_call/r1/transcript.md) |
| numerical_pilot | case_90 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_90/solo/r1/transcript.md) |
| numerical_pilot | case_91 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_91/debate/r1/transcript.md) |
| numerical_pilot | case_91 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_91/single_call/r1/transcript.md) |
| numerical_pilot | case_91 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_91/solo/r1/transcript.md) |
| numerical_pilot | case_92 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_92/debate/r1/transcript.md) |
| numerical_pilot | case_92 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_92/single_call/r1/transcript.md) |
| numerical_pilot | case_92 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_92/solo/r1/transcript.md) |
| numerical_pilot | case_93 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_93/debate/r1/transcript.md) |
| numerical_pilot | case_93 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_93/single_call/r1/transcript.md) |
| numerical_pilot | case_93 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_93/solo/r1/transcript.md) |
| numerical_pilot | case_94 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_94/debate/r1/transcript.md) |
| numerical_pilot | case_94 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_94/single_call/r1/transcript.md) |
| numerical_pilot | case_94 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_94/solo/r1/transcript.md) |
| numerical_pilot | case_95 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_95/debate/r1/transcript.md) |
| numerical_pilot | case_95 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_95/single_call/r1/transcript.md) |
| numerical_pilot | case_95 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_95/solo/r1/transcript.md) |
| numerical_pilot | case_96 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_96/debate/r1/transcript.md) |
| numerical_pilot | case_96 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_96/single_call/r1/transcript.md) |
| numerical_pilot | case_96 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / correct | [open](case_96/solo/r1/transcript.md) |
| numerical_pilot | case_97 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_97/debate/r1/transcript.md) |
| numerical_pilot | case_97 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_97/single_call/r1/transcript.md) |
| numerical_pilot | case_97 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_97/solo/r1/transcript.md) |
| numerical_pilot | case_98 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_98/debate/r1/transcript.md) |
| numerical_pilot | case_98 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | needs_more_evidence / abstention | [open](case_98/single_call/r1/transcript.md) |
| numerical_pilot | case_98 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_98/solo/r1/transcript.md) |
| numerical_pilot | case_99 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_99/debate/r1/transcript.md) |
| numerical_pilot | case_99 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_99/single_call/r1/transcript.md) |
| numerical_pilot | case_99 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_99/solo/r1/transcript.md) |
| numerical_pilot | case_100 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_100/debate/r1/transcript.md) |
| numerical_pilot | case_100 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | reject / wrong_verdict | [open](case_100/single_call/r1/transcript.md) |
| numerical_pilot | case_100 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_100/solo/r1/transcript.md) |
| numerical_pilot | case_101 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_101/debate/r1/transcript.md) |
| numerical_pilot | case_101 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_101/single_call/r1/transcript.md) |
| numerical_pilot | case_101 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_101/solo/r1/transcript.md) |
| numerical_pilot | case_102 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_102/debate/r1/transcript.md) |
| numerical_pilot | case_102 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_102/single_call/r1/transcript.md) |
| numerical_pilot | case_102 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_102/solo/r1/transcript.md) |
| numerical_pilot | case_103 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_103/debate/r1/transcript.md) |
| numerical_pilot | case_103 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_103/single_call/r1/transcript.md) |
| numerical_pilot | case_103 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_103/solo/r1/transcript.md) |
| numerical_pilot | case_104 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_104/debate/r1/transcript.md) |
| numerical_pilot | case_104 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_104/single_call/r1/transcript.md) |
| numerical_pilot | case_104 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_104/solo/r1/transcript.md) |
| numerical_pilot | case_105 | debate | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_105/debate/r1/transcript.md) |
| numerical_pilot | case_105 | single_call | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_105/single_call/r1/transcript.md) |
| numerical_pilot | case_105 | solo | r1 | accounts/fireworks/models/glm-5p3 / fireworks | completed | trust / correct | [open](case_105/solo/r1/transcript.md) |

468 recorded GLM trials. See [trace format](../TRACES.md) for provenance and capture limits.
