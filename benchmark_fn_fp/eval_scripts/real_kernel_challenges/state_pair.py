"""Private, independent GPU qualification for the state-update pair.

No evaluation model is invoked. Ground-truth declarations precede all tests.
The public implementation must already have been mounted under case_root.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


GROUND_TRUTH = {"case_106": "reject", "case_107": "trust"}


def _load(case_root: Path, case_id: str):
    spec = importlib.util.spec_from_file_location(
        f"state_qualification_{case_id}", case_root / case_id / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(shape, steps, pattern, seed):
    """Use exact representable data with transparent reproduction recipes."""
    b, length, width = shape
    if pattern == "zero_history":
        state = torch.zeros(shape, device="cuda", dtype=torch.float32)
        events = torch.arange(1, steps + 1, device="cuda", dtype=torch.float32)
        events = events[None, :, None].expand(b, steps, width).contiguous()
    elif pattern == "row_markers":
        # Distinct old rows expose duplication even far away from the append.
        rows = torch.arange(b * length, device="cuda", dtype=torch.float32)
        state = rows.reshape(b, length, 1).expand(b, length, width).contiguous()
        events = -torch.arange(1, b * steps + 1, device="cuda", dtype=torch.float32)
        events = events.reshape(b, steps, 1).expand(b, steps, width).contiguous()
    elif pattern == "signed_zero":
        state = torch.zeros(shape, device="cuda", dtype=torch.float32)
        state.view(torch.int32).reshape(-1)[::2] = -2147483648
        events = torch.zeros((b, steps, width), device="cuda", dtype=torch.float32)
        events.view(torch.int32).reshape(-1)[1::2] = -2147483648
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.randint(-4096, 4096, shape, generator=g).float().cuda()
        events = torch.randint(-4096, 4096, (b, steps, width), generator=g).float().cuda()
    return state, events


def _reference(state, events):
    # Deliberately independent of module.reference and the flat Triton indexing.
    result = torch.empty_like(state)
    count = events.shape[1]
    result[:, :state.shape[1] - count, :].copy_(state[:, count:, :])
    result[:, state.shape[1] - count:, :].copy_(events)
    return result


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _check(module, state, events):
    expected = _reference(state, events)
    public_expected = module.reference(state, events)
    if not _bits_equal(expected, public_expected):
        raise AssertionError("Independent and public references disagree")
    original_events = events.clone()
    output = module.run(state, events)
    torch.cuda.synchronize()
    difference = state.view(torch.int32) != expected.view(torch.int32)
    mismatches = int(difference.sum().item())
    same_storage = output.data_ptr() == state.data_ptr()
    events_preserved = _bits_equal(events, original_events)
    output_contract = (output.shape == state.shape and output.dtype == torch.float32
                       and output.device == state.device)
    passed = (mismatches == 0 and same_storage and events_preserved and output_contract)
    result = {"passed": passed, "mismatched_elements": mismatches,
              "same_storage": same_storage, "events_preserved": events_preserved,
              "output_contract": output_contract}
    if mismatches:
        # Flatten first: avoid materializing millions of coordinate triples.
        flat = difference.reshape(-1).to(torch.int32)
        first = int(torch.argmax(flat).item())
        width = state.shape[2]
        length = state.shape[1]
        result["first_mismatch"] = {
            "index": [first // (length * width), (first // width) % length, first % width],
            "expected": float(expected.reshape(-1)[first].item()),
            "actual": float(state.reshape(-1)[first].item()),
            "expected_int32_bits": int(expected.view(torch.int32).reshape(-1)[first].item()),
            "actual_int32_bits": int(state.view(torch.int32).reshape(-1)[first].item()),
        }
    return result


def validate(case_root: Path) -> dict:
    # These probes are fixed independently of model responses. All fit the public
    # domain. Many repetitions target nondeterministic execution, not relabeling.
    probes = [
        ((1, 2, 1), 2, "signed_zero", 3),
        ((1, 17, 37), 3, "random_integer", 5),
        ((2, 257, 65), 1, "row_markers", 16),
        ((4, 2048, 1024), 2, "zero_history", 64),
        ((4, 2048, 1024), 2, "row_markers", 64),
        ((2, 4096, 257), 1, "random_integer", 24),
        ((2, 1025, 1023), 127, "row_markers", 16),
        ((1, 128, 33), 128, "signed_zero", 3),
    ]
    cases = {}
    for case_id, label in GROUND_TRUTH.items():
        module = _load(case_root, case_id)
        records = []
        first_failure = None
        total_failures = 0
        total_runs = 0
        for shape, steps, pattern, repeats in probes:
            probe = {"shape": list(shape), "steps": steps, "pattern": pattern,
                     "repetitions": repeats, "seed": 314159,
                     "failing_repetitions": 0, "max_mismatched_elements": 0}
            for repeat in range(repeats):
                state, events = _inputs(shape, steps, pattern, 314159)
                checked = _check(module, state, events)
                total_runs += 1
                probe["max_mismatched_elements"] = max(
                    probe["max_mismatched_elements"], checked["mismatched_elements"])
                if not checked["passed"]:
                    total_failures += 1
                    probe["failing_repetitions"] += 1
                    if first_failure is None:
                        first_failure = dict(probe, repeat_index=repeat, evidence=checked)
                    if "first_failure" not in probe:
                        probe["first_failure"] = dict(repeat_index=repeat, **checked)
            records.append(probe)

        # Stateful semantics also apply to legal sequences of calls, rather than
        # only freshly initialized independent examples.
        state, _ = _inputs((2, 513, 129), 2, "row_markers", 314159)
        for sequence_step, steps in enumerate((1, 2, 17, 3, 128, 1)):
            _, events = _inputs((2, 513, 129), steps, "random_integer", 100 + sequence_step)
            checked = _check(module, state, events)
            total_runs += 1
            record = {"sequence_step": sequence_step, "shape": [2, 513, 129],
                      "steps": steps, "seed": 100 + sequence_step, **checked}
            records.append(record)
            if not checked["passed"]:
                total_failures += 1
                if first_failure is None:
                    first_failure = record

        qualified = total_failures > 0 if label == "reject" else total_failures == 0
        cases[case_id] = {
            "ground_truth": label,
            "qualification_passed": qualified,
            "total_runs": total_runs,
            "failing_runs": total_failures,
            "counterexample": first_failure,
            "tests": records,
            "label_scope": "entire public contract; fixed implementation, not a sampled seed",
            "evidence_limitation": (
                "A concrete legal mismatch qualifies rejection. Finite successful GPU runs "
                "support, but do not prove, acceptance; separate-storage source analysis is required."),
        }
    return {"cases": cases,
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
