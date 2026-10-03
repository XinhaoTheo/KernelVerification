"""Private qualification of the fixed segmented-recurrence implementations.

No model or GPU service is invoked by importing this file. validate(case_root)
runs on the caller's CUDA device; cpu_audit(case_root) is offline-only.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_112": "trust", "case_113": "reject"}


def plans():
    return [
        dict(shape=(1,1,1,1), chunk=16, pattern="none", style="random", seed=1),
        dict(shape=(1,15,1,7), chunk=16, pattern="inside", style="high", seed=2),
        dict(shape=(1,16,2,33), chunk=16, pattern="none", style="random", seed=3),
        dict(shape=(1,17,1,65), chunk=16, pattern="after", style="high", seed=4),
        dict(shape=(2,33,2,31), chunk=16, pattern="aligned", style="random", seed=5),
        dict(shape=(1,65,1,7), chunk=32, pattern="before", style="high", seed=6),
        dict(shape=(1,65,2,33), chunk=32, pattern="after", style="high", seed=7),
        dict(shape=(1,101,1,7), chunk=32, pattern="inside", style="isolation", seed=8),
        dict(shape=(2,101,3,65), chunk=32, pattern="inside", style="random", seed=9),
        dict(shape=(2,129,2,96), chunk=64, pattern="before", style="high", seed=10),
        dict(shape=(1,129,4,17), chunk=64, pattern="after", style="random", seed=11),
        dict(shape=(1,193,1,65), chunk=64, pattern="inside", style="isolation", seed=12),
        dict(shape=(2,257,4,96), chunk=16, pattern="none", style="high", seed=13),
        dict(shape=(2,257,2,33), chunk=32, pattern="many", style="alternating", seed=14),
        dict(shape=(1,67,2,17), chunk=64, pattern="singleton", style="random", seed=15),
        dict(shape=(2,97,2,33), chunk=32, pattern="gaps", style="random", seed=16),
        dict(shape=(1,97,2,33), chunk=16, pattern="inside", style="low", seed=17),
        dict(shape=(1,193,2,33), chunk=64, pattern="aligned", style="high", seed=18),
    ]


def arrays(plan):
    b, length, heads, width = plan["shape"]
    chunk = plan["chunk"]
    generator = np.random.default_rng(plan["seed"])
    u = generator.uniform(-1, 1, (b, length, heads, width)).astype(np.float32)
    decay = generator.uniform(.90, .96, (b, length, heads)).astype(np.float32)
    initial = generator.uniform(-1, 1, (b, heads, width)).astype(np.float32)
    labels = np.zeros((b, length), dtype=np.int32)
    pattern = plan["pattern"]
    for batch in range(b):
        if pattern == "none":
            boundaries = []
        elif pattern == "aligned":
            boundaries = list(range(chunk, length, chunk))
        elif pattern == "before":
            boundaries = [chunk - 1 + batch]
        elif pattern == "after":
            boundaries = [chunk + 1 + batch]
        elif pattern in {"inside", "gaps"}:
            boundaries = [min(chunk + chunk // 2 + batch, max(1, length // 2))]
        elif pattern == "many":
            boundaries = list(range(3 + batch, length, 7))
        elif pattern == "singleton":
            boundaries = list(range(1, length))
        else:
            raise ValueError(pattern)
        for boundary in boundaries:
            if 0 < boundary < length:
                labels[batch, boundary:] += 1
        if pattern == "gaps":
            labels[batch] *= 3
    style = plan["style"]
    if style in {"high", "isolation", "alternating"}:
        decay.fill(.96875)
    if style == "low":
        decay.fill(.5)
    if style == "high":
        u.fill(1)
        initial.fill(1)
    elif style == "isolation":
        u[:] = (labels == 0)[:, :, None, None]
        initial.fill(0)
    elif style == "alternating":
        u[:] = np.where(np.arange(length)[None, :, None, None] % 2, -1., 1.)
    return u, decay, labels, initial, chunk


def independent_reference(values):
    u, decay, labels, initial, _ = values
    b, length, heads, width = u.shape
    output = np.empty((b, length, heads, width), np.float64)
    state = initial.astype(np.float64).copy()
    for t in range(length):
        if t:
            state[labels[:, t] != labels[:, t-1]] = 0
        state = decay[:, t, :, None].astype(np.float64) * state + u[:, t].astype(np.float64)
        output[:, t] = state
    return output, state.copy()


def independent_by_segments(values):
    """Closed weighted sums per segment; no mutable cross-segment recurrence."""
    u, decay, labels, initial, _ = values
    result = np.zeros(u.shape, np.float64)
    for batch in range(u.shape[0]):
        starts = [0] + (np.nonzero(labels[batch, 1:] != labels[batch, :-1])[0] + 1).tolist()
        ends = starts[1:] + [u.shape[1]]
        for start, end in zip(starts, ends):
            for t in range(start, end):
                weight = np.ones(u.shape[2], np.float64)
                total = np.zeros((u.shape[2], u.shape[3]), np.float64)
                for s in range(t, start - 1, -1):
                    total += weight[:, None] * u[batch, s]
                    weight *= decay[batch, s]
                if start == 0:
                    total += weight[:, None] * initial[batch]
                result[batch, t] = total
    return result, result[:, -1].copy()


def chunk_decomposition(values, use_start_label=False):
    """Independent FP64 algebra check of the three-stage composition."""
    u, decay, labels, initial, chunk = values
    b, length, heads, width = u.shape
    count = (length + chunk - 1) // chunk
    local = np.empty_like(u, dtype=np.float64)
    prefix = np.empty_like(decay, dtype=np.float64)
    summary = np.zeros((b, count, heads, width), np.float64)
    scale = np.ones((b, count, heads), np.float64)
    for c in range(count):
        state = np.zeros((b, heads, width), np.float64)
        gain = np.ones((b, heads), np.float64)
        for t in range(c * chunk, min((c+1) * chunk, length)):
            if t > c * chunk:
                state[labels[:, t] != labels[:, t-1]] = 0
            state = decay[:, t, :, None] * state + u[:, t]
            gain *= decay[:, t]
            local[:, t] = state
            prefix[:, t] = gain
        summary[:, c] = state
        scale[:, c] = gain
    state = initial.astype(np.float64).copy()
    previous = np.zeros(b, np.int32)
    incoming = np.empty_like(summary)
    for c in range(count):
        incoming[:, c] = state
        representative = c * chunk if use_start_label else min((c+1) * chunk, length) - 1
        current = labels[:, representative]
        factor = np.where(current[:, None] == previous[:, None], scale[:, c], 0)
        state = factor[:, :, None] * state + summary[:, c]
        previous = current
    output = local.copy()
    for c in range(count):
        previous = labels[:, c * chunk - 1] if c else np.zeros(b, np.int32)
        for t in range(c * chunk, min((c+1) * chunk, length)):
            factor = np.where((labels[:, t] == previous)[:, None], prefix[:, t], 0)
            output[:, t] += factor[:, :, None] * incoming[:, c]
    return output, state


def _public_cpu_namespace(case_root):
    path = case_root / "case_112" / "kernel.py"
    names = {"validate_inputs", "reference", "make_inputs"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def cpu_audit(case_root: Path):
    public = _public_cpu_namespace(case_root)
    rows = []
    for plan in plans():
        value = arrays(plan)
        host = tuple(torch.from_numpy(v.copy()) for v in value[:-1]) + (value[-1],)
        public["validate_inputs"](*host, require_cuda=False)
        reference = independent_reference(value)
        separate = independent_by_segments(value)
        chunked = chunk_decomposition(value)
        public_ref = tuple(v.numpy() for v in public["reference"](*host))
        errors = {"independent_segments": max(np.max(np.abs(a-b)) for a,b in zip(reference,separate)),
                  "chunk_decomposition": max(np.max(np.abs(a-b)) for a,b in zip(reference,chunked)),
                  "public_reference": max(np.max(np.abs(a-b)) for a,b in zip(reference,public_ref))}
        assert max(errors.values()) < 1e-11, (plan, errors)
        candidate_error = max(np.max(np.abs(a-b)) for a,b in zip(reference,chunk_decomposition(value,True)))
        rows.append(dict(plan, agreement_errors={k:float(v) for k,v in errors.items()},
                         start_label_mutation_max_error=float(candidate_error)))
    isolation = arrays(plans()[7])
    expected = independent_reference(isolation)[0]
    assert np.max(np.abs(expected[isolation[2] != 0])) == 0
    invalid_checks = []
    for kind in ("decay_bound", "first_label", "nonmonotone", "grad", "negative_view",
                 "inference_tensor", "label_dtype", "overlap"):
        value = arrays(plans()[8])
        host = [torch.from_numpy(v.copy()) for v in value[:-1]]
        if kind == "decay_bound":
            host[1][0, 0, 0] = 1.0
        elif kind == "first_label":
            host[2][0, 0] = 1
        elif kind == "nonmonotone":
            host[2][0, 1] = 1
        elif kind == "grad":
            host[0].requires_grad_(True)
        elif kind == "negative_view":
            host[0] = torch._neg_view(host[0])
        elif kind == "inference_tensor":
            with torch.inference_mode():
                host[0] = host[0].clone()
        elif kind == "label_dtype":
            host[2] = host[2].long()
        elif kind == "overlap":
            host[0].fill_(.75)
            host[1] = host[0].reshape(-1)[:host[1].numel()].reshape(host[1].shape)
        try:
            public["validate_inputs"](*host, value[-1], require_cuda=False)
        except ValueError as exc:
            invalid_checks.append({"kind": kind, "correctly_rejected": True, "reason": str(exc)})
        else:
            raise AssertionError(f"Input validator accepted illegal {kind} case")
    return {"passed": True, "plans": rows,
            "invalid_input_checks": invalid_checks,
            "method": "FP64 serial vs independent per-segment weighted sums vs chunk decomposition vs public reference",
            "isolation_property": "When later-segment input is zero, earlier-segment signal has exactly zero later output in the independent reference."}


def _load(case_root, case):
    name = "segmented_qualification_" + case
    spec = importlib.util.spec_from_file_location(name, case_root / case / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root: Path):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, expected_label in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for plan in plans():
            value = arrays(plan)
            refs = independent_reference(value)
            gpu = tuple(torch.from_numpy(v.copy()).cuda() for v in value[:-1]) + (value[-1],)
            legality = module.validate_inputs(*gpu)
            before = tuple(v.clone() for v in gpu[:-1])
            repetitions = 3 if plan["style"] == "isolation" else 2
            worst = [0., 0.]
            abs_error = [0., 0.]
            first_witness = None
            finite, unchanged, types_valid = True, True, True
            for repeat in range(repetitions):
                output = module.run(*gpu)
                torch.cuda.synchronize()
                unchanged &= all(torch.equal(a,b) for a,b in zip(gpu[:-1],before))
                types_valid &= all(a.dtype == torch.float32 and a.device == gpu[0].device
                                   and tuple(a.shape) == tuple(r.shape) for a,r in zip(output,refs))
                for index, (actual, ref) in enumerate(zip(output,refs)):
                    actual_np = actual.double().cpu().numpy()
                    finite &= bool(np.isfinite(actual_np).all())
                    delta = np.abs(actual_np-ref)
                    ratios = delta / (.002 + .0001*np.abs(ref))
                    worst[index] = max(worst[index],float(ratios.max()))
                    abs_error[index] = max(abs_error[index],float(delta.max()))
                    if ratios.max() > 1 and first_witness is None:
                        location = np.unravel_index(np.argmax(ratios),ratios.shape)
                        first_witness = {"output": "states" if index == 0 else "final_state",
                            "index": list(map(int,location)), "reference": float(ref[location]),
                            "actual": float(actual_np[location]), "absolute_error": float(delta[location]),
                            "allowed_error": float(.002+.0001*abs(ref[location])),
                            "ratio": float(ratios[location]), "repeat": repeat}
            records.append(dict(plan, legal=legality["valid"], repetitions=repetitions,
                passed=bool(finite and unchanged and types_valid and max(worst)<=1),
                max_error_ratio=dict(zip(("states","final_state"),worst)),
                max_abs_error=dict(zip(("states","final_state"),abs_error)),
                finite=finite, inputs_preserved=unchanged, shape_dtype_valid=types_valid,
                witness=first_witness))
        failures = [row for row in records if not row["passed"]]
        qualified = not failures if expected_label == "trust" else bool(failures)
        cases[name] = {"ground_truth": expected_label, "qualification_passed": qualified,
                       "tests": records, "tests_total": len(records),
                       "tests_passed": len(records)-len(failures),
                       "counterexamples": failures,
                       "label_scope": "fixed implementation over the complete public input domain",
                       "analysis": "Stable bounded recurrence; input checks actually execute; CPU independent segmented and chunked algebra agree. Sampling is not a formal proof."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(c["qualification_passed"] for c in cases.values())}
