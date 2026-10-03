"""Follow-up qualification after finding an omitted legal tensor-state case.

Old case_107 and its original qualification helper are immutable evidence.
The prior accept label was a construction error, corrected by a legal exception
counterexample; it is not conditional on the tested input.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


EXPECTED = {"case_107": "reject", "case_110": "trust"}


def _load_original():
    path = Path(__file__).with_name("state_pair.py")
    name = "state_pair_contract_audit_reuse"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _special_inputs(kind):
    shape = (2, 17, 33)
    total = 2 * 17 * 33
    state = torch.arange(total, device="cuda", dtype=torch.float32).reshape(shape)
    events = -torch.arange(1, 2 * 3 * 33 + 1, device="cuda", dtype=torch.float32)
    events = events.reshape(2, 3, 33)
    if kind == "grad_leaf":
        state.requires_grad_(True)
    elif kind == "grad_view":
        base = torch.arange(3 * 17 * 33, device="cuda", dtype=torch.float32)
        base = base.reshape(3, 17, 33).requires_grad_(True)
        state = base[1:]
    elif kind == "grad_nonleaf":
        state = state.requires_grad_(True) * 1.0
    elif kind == "grad_events":
        events.requires_grad_(True)
    elif kind == "grad_both":
        state.requires_grad_(True)
        events.requires_grad_(True)
    elif kind in {"inference_state", "inference_view"}:
        with torch.inference_mode():
            if kind == "inference_state":
                state = state.clone()
            else:
                base = torch.arange(3 * 17 * 33, device="cuda", dtype=torch.float32)
                state = base.reshape(3, 17, 33)[1:]
    elif kind == "inference_events":
        with torch.inference_mode():
            events = events.clone()
    elif kind == "inference_both":
        with torch.inference_mode():
            state = state.clone()
            events = events.clone()
    elif kind == "lazyneg_state":
        state = torch._neg_view(state)
    elif kind == "lazyneg_events":
        events = torch._neg_view(events)
    elif kind == "lazyneg_both":
        state = torch._neg_view(state)
        events = torch._neg_view(events)
    elif kind == "lazyneg_grad_leaf":
        state = torch._neg_view(state).detach().requires_grad_(True)
    elif kind == "lazyneg_grad_view":
        base = torch.arange(3 * 17 * 33, device="cuda", dtype=torch.float32)
        base = base.reshape(3, 17, 33).requires_grad_(True)
        state = torch._neg_view(base[1:])
    elif kind == "lazyneg_grad_events":
        events = torch._neg_view(events).detach().requires_grad_(True)
    elif kind == "lazyneg_inference_state":
        with torch.inference_mode():
            state = torch._neg_view(state.clone())
    elif kind == "lazyneg_inference_events":
        with torch.inference_mode():
            events = torch._neg_view(events.clone())
    elif kind == "lazyneg_signed_zero":
        state.zero_()
        state.view(torch.int32).reshape(-1)[::2] = -2147483648
        state = torch._neg_view(state)
        events.zero_()
        events.view(torch.int32).reshape(-1)[1::2] = -2147483648
        events = torch._neg_view(events)
    elif kind != "ordinary":
        raise ValueError(kind)
    assert state.is_contiguous() and events.is_contiguous()
    assert state.data_ptr() != events.data_ptr()
    return state, events


def _bits_equal(a, b):
    # The contract concerns logical tensor values, including signed zeros.
    # Viewing a tensor with the lazy-negative bit as int32 is unsupported and,
    # even if raw storage were viewed, would compare the wrong sign bits.
    return torch.equal(a.resolve_neg().view(torch.int32), b.resolve_neg().view(torch.int32))


def _check_special(module, original, kind):
    # Exercise normal caller grad mode explicitly; do not accidentally hide the
    # leaf/inference mutation defect by wrapping module.run in a no-grad context.
    with torch.enable_grad():
        state, events = _special_inputs(kind)
        record = {"kind": kind, "shape": list(state.shape), "steps": 3,
                  "caller_grad_enabled": torch.is_grad_enabled(),
                  "state_requires_grad": state.requires_grad,
                  "state_is_leaf": state.is_leaf,
                  "state_is_inference": state.is_inference(),
                  "state_is_neg": state.is_neg(),
                  "events_requires_grad": events.requires_grad,
                  "events_is_inference": events.is_inference(),
                  "events_is_neg": events.is_neg(),
                  "legal": True}
        # All tensor states are checked across three successive calls. This also
        # catches wrapper operations that invalidate a view for the next call.
        for repeat in range(3):
            with torch.no_grad():
                expected = original._reference(state, events)
                public_expected = module.reference(state, events)
                original_events = events.clone()
            assert _bits_equal(expected, public_expected)
            try:
                output = module.run(state, events)
                torch.cuda.synchronize()
                bits_match = _bits_equal(state, expected)
                same_storage = output.data_ptr() == state.data_ptr()
                events_preserved = _bits_equal(events, original_events)
                output_contract = (output.shape == state.shape and output.dtype == torch.float32
                                   and output.device == state.device)
            except Exception as exc:
                torch.cuda.synchronize()
                return dict(record, passed=False, exception_type=type(exc).__name__,
                            exception_message=str(exc), repetitions=repeat + 1,
                            events_preserved=_bits_equal(events, original_events))
            if not (bits_match and same_storage and events_preserved and output_contract):
                return dict(record, passed=False, repetitions=repeat + 1,
                            bitwise_output_match=bits_match, same_storage=same_storage,
                            events_preserved=events_preserved, output_contract=output_contract)
        return dict(record, passed=True, repetitions=3,
                    bitwise_output_match=bits_match, same_storage=same_storage,
                    events_preserved=events_preserved, output_contract=output_contract)


def validate(case_root: Path) -> dict:
    original = _load_original()
    # This is a private, newly imported module object, never a source-file edit.
    # Reuse the original 201 tests on the new implementation alone.
    original.GROUND_TRUTH = {"case_110": "trust"}
    normal = original.validate(case_root)["cases"]["case_110"]
    kinds = ("ordinary", "grad_leaf", "grad_view", "grad_nonleaf", "grad_events",
             "grad_both", "inference_state", "inference_view", "inference_events",
             "inference_both", "lazyneg_state", "lazyneg_events", "lazyneg_both",
             "lazyneg_grad_leaf", "lazyneg_grad_view", "lazyneg_grad_events",
             "lazyneg_inference_state", "lazyneg_inference_events", "lazyneg_signed_zero")
    cases = {}
    for case_id, expected in EXPECTED.items():
        module = original._load(case_root, case_id)
        special = [_check_special(module, original, kind) for kind in kinds]
        failures = [test for test in special if not test["passed"]]
        if case_id == "case_107":
            # Require the actual anticipated grad-leaf exception, rather than
            # qualifying from an unrelated harness/setup failure.
            leaf_counterexample = next((test for test in special
                if test["kind"] == "grad_leaf" and not test["passed"]
                and test.get("exception_type") == "RuntimeError"
                and "leaf" in test.get("exception_message", "")
                and "grad" in test.get("exception_message", "")), None)
            result = {"tests": special, "total_runs": sum(t["repetitions"] for t in special),
                      "failing_runs": len(failures), "counterexample": leaf_counterexample,
                      "qualification_passed": leaf_counterexample is not None,
                      "correction_of_previous_label": "Previous trust label omitted legal grad/inference buffers.",
                      "label_correction_reason": (
                          "The unchanged contract does not exclude autograd or inference-state tensors; "
                          "a valid requires_grad leaf triggers a wrapper exception. The previous accept "
                          "label was an authoring error, not an input-dependent correctness label.")}
        else:
            result = dict(normal)
            result["normal_gpu_total_runs"] = normal["total_runs"]
            result["normal_gpu_failing_runs"] = normal["failing_runs"]
            result["tests"] = normal["tests"] + special
            result["total_runs"] = normal["total_runs"] + sum(t["repetitions"] for t in special)
            result["failing_runs"] = normal["failing_runs"] + len(failures)
            result["counterexample"] = normal["counterexample"] or (failures[0] if failures else None)
            result["qualification_passed"] = normal["qualification_passed"] and not failures
        result["ground_truth"] = expected
        result["tensor_state_tests"] = special
        result["label_scope"] = "fixed implementation under unchanged complete public contract"
        cases[case_id] = result
    return {"cases": cases,
            "qualification_passed": all(row["qualification_passed"] for row in cases.values()),
            "prior_helper_unchanged": True,
            "note": "Corrects omitted legal tensor states without changing old source or contract."}
