# Archived Input-Based Cases

Archived on October 19, 2026. **Cases 103–150 are excluded from the current benchmark, default scoring and new agent runs.** The current benchmark has 102 cases. See the [current case index](../../CASE_INDEX.md) and [Tools vs No Tools](../../SUCCESS_CASES.md).

These 48 cases assigned different labels to the same computation by changing selected data, fixed parameters or the permitted input domain. They do not meet the current design rule: compare implementation correctness over a shared complete input domain. A test input can expose a defect, but must not define a separate label for the same kernel.

| Cases | Reason for exclusion | Historical material |
|---|---|---|
| 103–104 | Identical atomic-sum code; changing the input-value domain changes the label. | [Public cases](triton_eval_cases/) · [Original answers](triton/) |
| 105–130 | Selected tensors, seeds or permutations determine whether the same approximate computation passes. | [Numerical studies](single_call_vs_tools_challenges/README.md) |
| 131–140 | Selected regression inputs, cache permutations, dropout tables, alignment data or optimizer updates determine the label. | [Evidence studies](solo_vs_debate_challenges/README.md) |
| 141–142 | The requirement covers all legal X, but selected fixed parameter inputs change the target and its error bound. The computation is unchanged. | [Case 141](triton_eval_cases/case_141/problem.txt) · [Case 142](triton_eval_cases/case_142/problem.txt) |
| 143–150 | The same summation implementation receives different labels for selected rows and orders. | [Evidence studies](solo_vs_debate_challenges/README.md) |

The archive contains **299 GLM trials and 11 Opus trials**. Public kernels, problem descriptions, labels, raw requests, responses and probe records retain their original bytes. Documentation links have been updated for the move. These are historical results; moving them does not mean that an experiment was repeated or a verdict was changed.

| Content | Location |
|---|---|
| Public kernel and problem files | [triton_eval_cases](triton_eval_cases/) |
| GLM experiment records | [traces_glm](traces_glm/) |
| Opus experiment records | [traces_opus5](traces_opus5/) |
| Fixed-input numerical study and private records | [single_call_vs_tools_challenges](single_call_vs_tools_challenges/) |
| Initial-evidence study and private records | [solo_vs_debate_challenges](solo_vs_debate_challenges/) |
| Case identity, archived status and current paths | [case_map.json](../../case_map.json) |

The older [24-case numerical pilot](../numerical_pilot/README.md), cases 151–174, remains archived separately. Its fixed workloads are also excluded. The current benchmark uses cases 01–102. Original identifiers remain in raw records and in the registry mapping.

To inspect historical scores without changing the current scoreboard, run `python benchmark_fn_fp/eval_scripts/summarize_traces.py --include-archived --json`. Historical author scripts remain under `eval_scripts/`; their default paths now point into this archive.
