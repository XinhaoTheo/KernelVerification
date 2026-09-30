# case_50–case_61 construction and evaluation protocol

Storage update (2026-09-30): Batch labels and repeat shorthand below retain their original experimental meaning; original labels are saved in `trace_meta.json.original_trial`. Canonical links use per-case/arm `rN` folders; see [trace naming and migration](../TRACES.md).

Recorded before any case_50–case_61 model evaluation. Twelve additional fixed-workload
cases form six balanced pairs, with one numerically compliant and one
noncompliant case per pair. They extend case_36–case_49 without editing prior cases,
protocols, or recorded outcomes.

Planned mechanisms: orthogonal projection and normalization (case_50/case_51), quantized
nearest-neighbor routing (case_52/case_53), under-resolved quadrature (case_54/case_55), Fourier band
truncation (case_56/case_57), finite-precision log determinant (case_58/case_59), and expanded squared
distances followed by an RBF transform (case_60/case_61). Final construction details and
all CPU calibration/search records are retained in the family modules and
search logs. Labels are selected using numerical calculations, without LLM
feedback. Each public pair shares its source/contract apart from an input seed.
The reference always uses the actual stored FP32 inputs.

Before evaluation, compare two independently implemented numerical references,
validate the real Triton kernel ten times on T4, and freeze source/input hashes.
Use a margin of at least 25% from the threshold in CPU selection and confirm
the GPU labels. An unevaluated construction may be repaired if GPU compilation
or truth validation fails; preserve the failed validation log. Never modify a
case after its first model evaluation. Private oracle/search/answer files are
not mounted in the agentic evaluation image.

All arms use accounts/fireworks/models/glm-5p3 through the existing Fireworks
credential, with the original prompts. Matched settings are reasoning_effort
low and a 32768-token cap per call. Solo permits ten rounds; the existing
four-role debate permits four rounds. These are not matched total compute or
total cost budgets.

Predeclared trials, covering all twelve cases regardless of initial outcome:

- oz_low32_r1: single_call, solo, debate.
- oz_low32_r2: single_call, solo, debate.
- oz_low32_r3: single_call only.
- oz_default64_r1: single_call only, omit reasoning_effort, 65536-token cap.
  This separate control tests configuration sensitivity. It is not included
  in the matched low-setting aggregate or replication gate.

This is 84 low-setting experimental runs plus 12 default-reasoning controls;
tool runs may contain multiple API calls. All four cohorts are reported, even
when they do not show an advantage. No wrong answer triggers an unrecorded retry.
Failures, abstentions, and token exhaustion are separate from explicit mistakes.
No late retry replaces a designated failed slot. Keep all attempts and provider
errors, including calls that returned no usage and therefore have unknown cost.

The exploratory target is at least three additional mechanisms each containing
a case with two or more explicit source-only mistakes in its three designated
low-setting trials and both tool arms correct in both designated trials. Every
qualifying slot needs a complete trace and terminal, explicit verdict. This is
replication on selected synthetic workloads, not held-out generalization.
Default-control counterevidence and transport failures remain visible.

Use the unified ../traces_glm/<case>/<arm>/rN/ layout. Save requests before
calls, all returned provider responses/reasoning/usage, errors and finish reasons,
readable transcripts, and all tool probes and results. Report source-only, solo,
and debate separately, and audit actual runtime evidence before attributing a
tool verdict to successful verification. Costs are project profile estimates,
excluding Modal GPU charges and any unreported usage. case_50–case_61 results are generated
in OZ_REPORT.md / private_data/reports/oz_scoreboard.json without rewriting the case_44–case_49 result files.

## Construction-record scope clarification

The six final-configuration formal seed sweeps are completely retained, along
with the calibration rows counted in OZ_DESIGN.md. During the subsequent
documentation audit, we found that early case_58/case_59 regularizer and case_60/case_61 offset/target
experiments did not retain every candidate row. Available aggregate summaries
were archived afterward in OZ_EARLY_PARAMETER_EXPLORATION.md with an explicit
archive time. They are not reconstructed original row-level logs. Thus the
record is complete for formal seed selection and all model attempts, but not
for every preliminary numerical parameter exploration. This clarification does
not change any frozen case, model configuration, designated trial, or success
criterion above.

报告默认只写 Markdown。`private_data/reports/` 中的 JSON 是可选缓存，需运行对应报告脚本并传入 `--json` 才会生成；清理时删除这些缓存不会删除实验记录。
