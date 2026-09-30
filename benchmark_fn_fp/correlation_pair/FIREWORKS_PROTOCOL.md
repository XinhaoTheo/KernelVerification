# Fireworks replication: frozen before calls

Run date: 2026-09-22. Model: accounts/fireworks/models/glm-5p3, provider Fireworks, FIREWORKS_API_KEY.

Six planned runs: case_36 and case_37, each single_call, solo, debate. One trial per cell; no case selection or threshold changes. Original contract wording, no post-hoc wording ablation. Frozen GPU oracle: validation_gpu.json. Preserve original Opus traces; new traces_fireworks tree.

Single-call has no tools, original baseline system/user prompt, identical verdict schema appended explicitly and JSON-object response format. Preserve full provider response (including finish reason and reasoning when supplied). Solo and debate use the existing verifier prompts and tools on T4. Debate roles: describer, skeptic, experimenter, judge. Same model and max_tokens=32768 per call in all arms. Solo max rounds=10; debate max rounds=4 (existing settings), so these are practical configurations, not matched total inference budgets. Timeout=1800 seconds per API call. Existing SDK retries for tool arms; single-call retries=0. No semantic retry or cherry-picking.

Record verdict correctness separately from abstention, malformed output, token exhaustion, infrastructure errors. Compare grounded numerical evidence, probes, turns, input/output tokens and estimated API cost from the existing model profile ($0.28/$1.10 per million input/output tokens), excluding Modal charges. This price is a project-config estimate, not a verified current invoice. Two selected cases cannot establish statistical or general debate superiority. If solo and debate both solve both cases, the result does not demonstrate extra four-role accuracy benefit.

Scheduling note: after both solo trials finished, debate B was launched in a separate Modal app while debate A remained in progress. The original sequential driver refuses an existing B trial directory before any remote call, preventing a duplicate B run. A resulting local `Trace already exists` exit is a scheduling guard, not a model failure or extra trial.

Startup note: the initial local source-only command failed at import (`openai` missing), before making any API call. It was rerun using an already-present isolated environment with the dependency installed. The import error is preserved in fireworks_startup_error.txt.

## Reproduction (fresh trial name required)

```sh
uv run --with openai --with python-dotenv python benchmark_fn_fp/correlation_pair/run_fireworks_single.py --trial r2
modal run benchmark_fn_fp/correlation_pair/evaluate_modal.py --action solo --provider fireworks --model accounts/fireworks/models/glm-5p3 --trial r2
modal run benchmark_fn_fp/correlation_pair/evaluate_modal.py --action debate --provider fireworks --model accounts/fireworks/models/glm-5p3 --trial r2
python benchmark_fn_fp/correlation_pair/report.py --fireworks
```

These commands make new paid calls; `r2` above is a reproduction example, not an executed trial in this experiment. Existing trial directories cannot be overwritten. The Modal image contains only verifier code and evaluation cases; construction logs, labels and oracle measurements are not mounted into the agent environment.

Latency is summed wall time around SDK calls, including provider/network waiting and SDK retries where enabled; it is not pure generation time and is not sufficient to causally attribute slowness to role structure. API estimates use returned token usage only; unreported usage from failed/retried HTTP requests and Modal GPU charges are not measured here.
