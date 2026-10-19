# Allclose results

October 19, 2026. NVIDIA T4, PyTorch 2.8.0, Triton 3.4.0. No model API calls.

| Set | Correct | Wrong | No verdict | Accuracy |
|---|---:|---:|---:|---:|
| Original cases 01–32 | 1/32 | 31 | 0 | 3.1% |
| Kernel challenges 33–90 | 48/58 | 10 | 0 | 82.8% |
| All active cases | 49/90 | 41 | 0 | 54.4% |

The original cases execute their unchanged `test.py` and read `NAIVE_ALLCLOSE_VERDICT`. These checks intentionally include limitations such as fixed tolerances, one output layout or independent random samples. Their low score describes this selected suite.

The kernel challenges use one public `make_inputs(device="cuda", seed=0)` call with all other defaults unchanged. The public reference is computed before the candidate runs, because some kernels change their inputs. Comparisons use the stated numerical tolerances, exact integer values or exact bit patterns. They check numerical outputs, not every legal value, shape, alias or storage rule. No private qualification inputs or saved failing examples are used. The public erf reference rounds to FP32; this is a numerical baseline, not a mathematical error proof.

A passing test maps to Accept; a mismatch maps to Reject. Exceptions map to no verdict. The kernel ground-truth label stays fixed for its full problem description. The two sets use different comparison protocols; the aggregate is a history summary, not one universal allclose setting.

The [r2 manifest](manifest_r2.json) and [saved harness](protocol_r2/) were saved before execution. [r2 summary](summary_r2.json) contains all 90 results. [r1](startup_r1.json) failed during container import and was stopped before any kernel ran; all cancelled calls remain recorded. The local Modal client printed a cleanup warning after collecting r2; all 90 result files report complete checks, with no missing result.

## Cost

Model API cost: **$0**. Completed checks took 62.67 summed GPU-worker seconds; the collection took 32.64 wall-clock seconds. The two Modal apps lived for 21 and 37 seconds, with at most three T4 workers. At [Modal published rates](https://modal.com/pricing), assuming three workers and their requested CPU/memory for both full lifetimes gives about **$0.04**, including the failed startup. This is an estimate, not an invoice; CPU/memory usage and billing adjustments can change it. [Calculation and app IDs](cost_r2.json).

## Cases

Previous IDs are shown to match older notes and raw model transcripts. All paths use the new numbering.

| Case | Previous ID | Ground truth | Allclose | Correct | Record |
|---|---|---|---|---|---|
| case_01 | case_01 | Reject | Accept | No | [result](case_01/r2/result.json) · [stdout](case_01/r2/stdout.txt) |
| case_02 | case_02 | Accept | Reject | No | [result](case_02/r2/result.json) · [stdout](case_02/r2/stdout.txt) |
| case_03 | case_04 | Accept | Accept | Yes | [result](case_03/r2/result.json) · [stdout](case_03/r2/stdout.txt) |
| case_04 | case_05 | Accept | Reject | No | [result](case_04/r2/result.json) · [stdout](case_04/r2/stdout.txt) |
| case_05 | case_06 | Reject | Accept | No | [result](case_05/r2/result.json) · [stdout](case_05/r2/stdout.txt) |
| case_06 | case_07 | Accept | Reject | No | [result](case_06/r2/result.json) · [stdout](case_06/r2/stdout.txt) |
| case_07 | case_08 | Accept | Reject | No | [result](case_07/r2/result.json) · [stdout](case_07/r2/stdout.txt) |
| case_08 | case_09 | Reject | Accept | No | [result](case_08/r2/result.json) · [stdout](case_08/r2/stdout.txt) |
| case_09 | case_10 | Reject | Accept | No | [result](case_09/r2/result.json) · [stdout](case_09/r2/stdout.txt) |
| case_10 | case_11 | Accept | Reject | No | [result](case_10/r2/result.json) · [stdout](case_10/r2/stdout.txt) |
| case_11 | case_12 | Reject | Accept | No | [result](case_11/r2/result.json) · [stdout](case_11/r2/stdout.txt) |
| case_12 | case_13 | Reject | Accept | No | [result](case_12/r2/result.json) · [stdout](case_12/r2/stdout.txt) |
| case_13 | case_14 | Reject | Accept | No | [result](case_13/r2/result.json) · [stdout](case_13/r2/stdout.txt) |
| case_14 | case_15 | Reject | Accept | No | [result](case_14/r2/result.json) · [stdout](case_14/r2/stdout.txt) |
| case_15 | case_16 | Reject | Accept | No | [result](case_15/r2/result.json) · [stdout](case_15/r2/stdout.txt) |
| case_16 | case_17 | Reject | Accept | No | [result](case_16/r2/result.json) · [stdout](case_16/r2/stdout.txt) |
| case_17 | case_18 | Reject | Accept | No | [result](case_17/r2/result.json) · [stdout](case_17/r2/stdout.txt) |
| case_18 | case_19 | Reject | Accept | No | [result](case_18/r2/result.json) · [stdout](case_18/r2/stdout.txt) |
| case_19 | case_20 | Reject | Accept | No | [result](case_19/r2/result.json) · [stdout](case_19/r2/stdout.txt) |
| case_20 | case_21 | Reject | Accept | No | [result](case_20/r2/result.json) · [stdout](case_20/r2/stdout.txt) |
| case_21 | case_22 | Accept | Reject | No | [result](case_21/r2/result.json) · [stdout](case_21/r2/stdout.txt) |
| case_22 | case_23 | Accept | Reject | No | [result](case_22/r2/result.json) · [stdout](case_22/r2/stdout.txt) |
| case_23 | case_24 | Accept | Reject | No | [result](case_23/r2/result.json) · [stdout](case_23/r2/stdout.txt) |
| case_24 | case_25 | Reject | Accept | No | [result](case_24/r2/result.json) · [stdout](case_24/r2/stdout.txt) |
| case_25 | case_26 | Accept | Reject | No | [result](case_25/r2/result.json) · [stdout](case_25/r2/stdout.txt) |
| case_26 | case_27 | Reject | Accept | No | [result](case_26/r2/result.json) · [stdout](case_26/r2/stdout.txt) |
| case_27 | case_28 | Reject | Accept | No | [result](case_27/r2/result.json) · [stdout](case_27/r2/stdout.txt) |
| case_28 | case_29 | Accept | Reject | No | [result](case_28/r2/result.json) · [stdout](case_28/r2/stdout.txt) |
| case_29 | case_30 | Reject | Accept | No | [result](case_29/r2/result.json) · [stdout](case_29/r2/stdout.txt) |
| case_30 | case_31 | Reject | Accept | No | [result](case_30/r2/result.json) · [stdout](case_30/r2/stdout.txt) |
| case_31 | case_32 | Reject | Accept | No | [result](case_31/r2/result.json) · [stdout](case_31/r2/stdout.txt) |
| case_32 | case_33 | Reject | Accept | No | [result](case_32/r2/result.json) · [stdout](case_32/r2/stdout.txt) |
| case_33 | case_106 | Reject | Reject | Yes | [result](case_33/r2/result.json) · [stdout](case_33/r2/stdout.txt) |
| case_34 | case_107 | Reject | Accept | No | [result](case_34/r2/result.json) · [stdout](case_34/r2/stdout.txt) |
| case_35 | case_108 | Reject | Accept | No | [result](case_35/r2/result.json) · [stdout](case_35/r2/stdout.txt) |
| case_36 | case_109 | Reject | Reject | Yes | [result](case_36/r2/result.json) · [stdout](case_36/r2/stdout.txt) |
| case_37 | case_110 | Accept | Accept | Yes | [result](case_37/r2/result.json) · [stdout](case_37/r2/stdout.txt) |
| case_38 | case_111 | Accept | Accept | Yes | [result](case_38/r2/result.json) · [stdout](case_38/r2/stdout.txt) |
| case_39 | case_112 | Accept | Accept | Yes | [result](case_39/r2/result.json) · [stdout](case_39/r2/stdout.txt) |
| case_40 | case_113 | Reject | Accept | No | [result](case_40/r2/result.json) · [stdout](case_40/r2/stdout.txt) |
| case_41 | case_114 | Reject | Accept | No | [result](case_41/r2/result.json) · [stdout](case_41/r2/stdout.txt) |
| case_42 | case_115 | Accept | Accept | Yes | [result](case_42/r2/result.json) · [stdout](case_42/r2/stdout.txt) |
| case_43 | case_116 | Reject | Accept | No | [result](case_43/r2/result.json) · [stdout](case_43/r2/stdout.txt) |
| case_44 | case_117 | Accept | Accept | Yes | [result](case_44/r2/result.json) · [stdout](case_44/r2/stdout.txt) |
| case_45 | case_118 | Accept | Accept | Yes | [result](case_45/r2/result.json) · [stdout](case_45/r2/stdout.txt) |
| case_46 | case_119 | Reject | Reject | Yes | [result](case_46/r2/result.json) · [stdout](case_46/r2/stdout.txt) |
| case_47 | case_120 | Accept | Accept | Yes | [result](case_47/r2/result.json) · [stdout](case_47/r2/stdout.txt) |
| case_48 | case_121 | Reject | Reject | Yes | [result](case_48/r2/result.json) · [stdout](case_48/r2/stdout.txt) |
| case_49 | case_122 | Accept | Accept | Yes | [result](case_49/r2/result.json) · [stdout](case_49/r2/stdout.txt) |
| case_50 | case_123 | Reject | Reject | Yes | [result](case_50/r2/result.json) · [stdout](case_50/r2/stdout.txt) |
| case_51 | case_124 | Accept | Accept | Yes | [result](case_51/r2/result.json) · [stdout](case_51/r2/stdout.txt) |
| case_52 | case_125 | Reject | Accept | No | [result](case_52/r2/result.json) · [stdout](case_52/r2/stdout.txt) |
| case_53 | case_126 | Accept | Accept | Yes | [result](case_53/r2/result.json) · [stdout](case_53/r2/stdout.txt) |
| case_54 | case_127 | Reject | Reject | Yes | [result](case_54/r2/result.json) · [stdout](case_54/r2/stdout.txt) |
| case_55 | case_128 | Accept | Accept | Yes | [result](case_55/r2/result.json) · [stdout](case_55/r2/stdout.txt) |
| case_56 | case_129 | Reject | Reject | Yes | [result](case_56/r2/result.json) · [stdout](case_56/r2/stdout.txt) |
| case_57 | case_130 | Accept | Accept | Yes | [result](case_57/r2/result.json) · [stdout](case_57/r2/stdout.txt) |
| case_58 | case_131 | Reject | Reject | Yes | [result](case_58/r2/result.json) · [stdout](case_58/r2/stdout.txt) |
| case_59 | case_132 | Accept | Accept | Yes | [result](case_59/r2/result.json) · [stdout](case_59/r2/stdout.txt) |
| case_60 | case_133 | Reject | Accept | No | [result](case_60/r2/result.json) · [stdout](case_60/r2/stdout.txt) |
| case_61 | case_134 | Accept | Accept | Yes | [result](case_61/r2/result.json) · [stdout](case_61/r2/stdout.txt) |
| case_62 | case_135 | Reject | Reject | Yes | [result](case_62/r2/result.json) · [stdout](case_62/r2/stdout.txt) |
| case_63 | case_136 | Accept | Accept | Yes | [result](case_63/r2/result.json) · [stdout](case_63/r2/stdout.txt) |
| case_64 | case_137 | Reject | Reject | Yes | [result](case_64/r2/result.json) · [stdout](case_64/r2/stdout.txt) |
| case_65 | case_138 | Accept | Accept | Yes | [result](case_65/r2/result.json) · [stdout](case_65/r2/stdout.txt) |
| case_66 | case_139 | Reject | Reject | Yes | [result](case_66/r2/result.json) · [stdout](case_66/r2/stdout.txt) |
| case_67 | case_140 | Accept | Accept | Yes | [result](case_67/r2/result.json) · [stdout](case_67/r2/stdout.txt) |
| case_68 | case_141 | Reject | Reject | Yes | [result](case_68/r2/result.json) · [stdout](case_68/r2/stdout.txt) |
| case_69 | case_142 | Accept | Accept | Yes | [result](case_69/r2/result.json) · [stdout](case_69/r2/stdout.txt) |
| case_70 | case_143 | Reject | Reject | Yes | [result](case_70/r2/result.json) · [stdout](case_70/r2/stdout.txt) |
| case_71 | case_144 | Accept | Accept | Yes | [result](case_71/r2/result.json) · [stdout](case_71/r2/stdout.txt) |
| case_72 | case_145 | Reject | Accept | No | [result](case_72/r2/result.json) · [stdout](case_72/r2/stdout.txt) |
| case_73 | case_146 | Accept | Accept | Yes | [result](case_73/r2/result.json) · [stdout](case_73/r2/stdout.txt) |
| case_74 | case_147 | Reject | Reject | Yes | [result](case_74/r2/result.json) · [stdout](case_74/r2/stdout.txt) |
| case_75 | case_148 | Accept | Accept | Yes | [result](case_75/r2/result.json) · [stdout](case_75/r2/stdout.txt) |
| case_76 | case_149 | Reject | Accept | No | [result](case_76/r2/result.json) · [stdout](case_76/r2/stdout.txt) |
| case_77 | case_150 | Accept | Accept | Yes | [result](case_77/r2/result.json) · [stdout](case_77/r2/stdout.txt) |
| case_78 | case_151 | Reject | Reject | Yes | [result](case_78/r2/result.json) · [stdout](case_78/r2/stdout.txt) |
| case_79 | case_152 | Accept | Accept | Yes | [result](case_79/r2/result.json) · [stdout](case_79/r2/stdout.txt) |
| case_80 | case_153 | Reject | Reject | Yes | [result](case_80/r2/result.json) · [stdout](case_80/r2/stdout.txt) |
| case_81 | case_154 | Accept | Accept | Yes | [result](case_81/r2/result.json) · [stdout](case_81/r2/stdout.txt) |
| case_82 | case_155 | Reject | Reject | Yes | [result](case_82/r2/result.json) · [stdout](case_82/r2/stdout.txt) |
| case_83 | case_156 | Accept | Accept | Yes | [result](case_83/r2/result.json) · [stdout](case_83/r2/stdout.txt) |
| case_84 | case_157 | Reject | Accept | No | [result](case_84/r2/result.json) · [stdout](case_84/r2/stdout.txt) |
| case_85 | case_158 | Accept | Accept | Yes | [result](case_85/r2/result.json) · [stdout](case_85/r2/stdout.txt) |
| case_86 | case_159 | Reject | Reject | Yes | [result](case_86/r2/result.json) · [stdout](case_86/r2/stdout.txt) |
| case_87 | case_160 | Accept | Accept | Yes | [result](case_87/r2/result.json) · [stdout](case_87/r2/stdout.txt) |
| case_88 | case_161 | Reject | Reject | Yes | [result](case_88/r2/result.json) · [stdout](case_88/r2/stdout.txt) |
| case_89 | case_162 | Accept | Accept | Yes | [result](case_89/r2/result.json) · [stdout](case_89/r2/stdout.txt) |
| case_90 | case_163 | Reject | Reject | Yes | [result](case_90/r2/result.json) · [stdout](case_90/r2/stdout.txt) |
