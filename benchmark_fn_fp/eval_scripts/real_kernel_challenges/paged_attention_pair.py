"""Private independent qualification for case_114/115; no model calls.

Historical sliding-window value masking mechanism from vLLM PR #30887.
The fixed expected labels refer to whole public contracts, never input seeds.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

EXPECTED = {"case_114": "reject", "case_115": "trust"}
GROUND_TRUTH = EXPECTED


def _load(case_root, case):
    spec = importlib.util.spec_from_file_location(
        "paged_qualification_" + case, Path(case_root) / case / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(config, poison="finite", seed=713):
    """Construct CPU inputs independently of public make_inputs()."""
    import torch
    b, qt, hk, group, d, page, lengths, window, mapping, pattern = config
    generator = torch.Generator().manual_seed(seed)
    cap = (max(lengths) + page - 1) // page
    pages = b * cap + 3
    order = torch.randperm(pages, generator=generator)
    if mapping == "reverse":
        order = torch.arange(pages - 1, -1, -1)
    elif mapping == "identity":
        order = torch.arange(pages)
    table = order[:b * cap].reshape(b, cap).to(torch.int32).contiguous()
    if mapping == "repeat":
        for batch in range(b):
            count = (lengths[batch] + page - 1) // page
            if count > 1:
                table[batch, count - 1] = table[batch, 0]
    elif mapping == "cross_batch":
        for batch in range(1, b):
            table[batch] = table[0]
    q = (2 * torch.rand((b, qt, hk * group, d), generator=generator) - 1).half()
    if pattern == "uniform_scores":
        q.zero_()
    elif pattern == "signed_extremes":
        q = torch.where(q >= 0, 1.0, -1.0).half()
    shape = (pages, page, hk, d)
    if poison == "finite":
        k = (2 * torch.rand(shape, generator=generator) - 1).half()
        v = (2 * torch.rand(shape, generator=generator) - 1).half()
    else:
        k = torch.full(shape, float("nan"), dtype=torch.float16)
        v = torch.full(shape, float("nan") if poison == "nan" else float("inf"),
                       dtype=torch.float16)
        if poison == "infinity":
            v.reshape(-1)[::2] = float("-inf")
    for batch, length in enumerate(lengths):
        lo = max(0, length - qt - window + 1) if window else 0
        for position in range(lo, length):
            physical = int(table[batch, position // page])
            kval = (2 * torch.rand((hk, d), generator=generator) - 1).half()
            vval = (2 * torch.rand((hk, d), generator=generator) - 1).half()
            if pattern == "uniform_scores":
                vval.fill_(((position % 31) - 15) / 16.0)
            elif pattern == "signed_extremes":
                kval = torch.where(kval >= 0, 1.0, -1.0).half()
                vval = torch.where(vval >= 0, 1.0, -1.0).half()
            k[physical, position % page] = kval
            v[physical, position % page] = vval
        # Slots after the logical page count are irrelevant, even in block_table.
        count = (length + page - 1) // page
        table[batch, count::2] = -2147483648
        table[batch, count + 1::2] = 2147483647
    return q, k, v, table, torch.tensor(lengths, dtype=torch.int32), window


def independent_reference(values):
    """NumPy FP64 softmax over gathered logical positions, no Triton helpers."""
    import numpy as np
    q, k, v, table, lengths, window = values
    q, k, v, table, lengths = (t.cpu().numpy() for t in (q, k, v, table, lengths))
    batch_count, qt, hq, d = q.shape
    page, hk = k.shape[1:3]
    result = np.empty(q.shape, dtype=np.float64)
    for batch in range(batch_count):
        for row in range(qt):
            end = int(lengths[batch]) - qt + row + 1
            start = max(0, end - window) if window else 0
            positions = np.arange(start, end)
            physical = table[batch, positions // page]
            for head in range(hq):
                kv = head // (hq // hk)
                keys = k[physical, positions % page, kv].astype(np.float64)
                vals = v[physical, positions % page, kv].astype(np.float64)
                scores = (keys * q[batch, row, head].astype(np.float64)).sum(1) / np.sqrt(d)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                result[batch, row, head] = (probabilities[:, None] * vals).sum(0)
    return result


def independently_legal(values):
    """Check construction legality independently of the public validator."""
    import numpy as np
    q, k, v, table, lengths, window = values
    if not np.isfinite(q.numpy()).all() or float(q.abs().max()) > 1:
        return False
    for batch, length in enumerate(lengths.tolist()):
        if not (q.shape[1] <= length <= 512):
            return False
        page = k.shape[1]
        count = (length + page - 1) // page
        ids = table[batch, :count]
        if not bool(((ids >= 0) & (ids < k.shape[0])).all()):
            return False
        lo = max(0, length - q.shape[1] - window + 1) if window else 0
        for logical in range(lo, length):
            physical = int(table[batch, logical // page])
            for cache in (k, v):
                row = cache[physical, logical % page].numpy()
                if not np.isfinite(row).all() or np.max(np.abs(row)) > 1:
                    return False
    return True


def plans():
    # B, Q, KV heads, grouped-query ratio, D, page size, lengths, window,
    # physical mapping policy, active-value pattern.
    return [
        (1, 1, 1, 4, 64, 16, [79], 19, "permute", "uniform_scores"),
        (1, 1, 1, 1, 32, 16, [1], 1, "identity", "random"),
        (1, 7, 2, 2, 64, 16, [97], 41, "reverse", "random"),
        (2, 9, 2, 4, 64, 32, [137, 79], 65, "permute", "random"),
        (1, 17, 1, 8, 128, 16, [255], 63, "permute", "signed_extremes"),
        (2, 33, 2, 1, 32, 64, [512, 257], 127, "reverse", "random"),
        (4, 3, 4, 4, 32, 16, [65, 128, 257, 512], 1, "permute", "random"),
        (1, 16, 1, 1, 64, 16, [80], 17, "permute", "uniform_scores"),
        (2, 17, 1, 2, 128, 32, [65, 95], 32, "repeat", "random"),
        (1, 33, 2, 2, 64, 64, [511], 256, "permute", "random"),
        (2, 15, 1, 4, 32, 16, [15, 127], 0, "permute", "random"),
        (1, 32, 2, 8, 128, 32, [512], 0, "reverse", "signed_extremes"),
        (1, 2, 1, 2, 64, 16, [128], 64, "permute", "random"),
        (1, 2, 1, 2, 64, 16, [129], 64, "permute", "random"),
        (2, 5, 2, 4, 64, 16, [65, 95], 17, "cross_batch", "random"),
        (2, 4, 1, 4, 128, 32, [4, 129], 1, "cross_batch", "signed_extremes"),
        (1, 8, 1, 2, 64, 16, [8], 0, "permute", "signed_extremes"),
        (1, 33, 1, 8, 128, 16, [33], 1, "permute", "signed_extremes"),
    ]


def validate(case_root: Path) -> dict:
    import numpy as np
    import torch
    cases = {}
    for case_id, truth in EXPECTED.items():
        candidate = _load(case_root, case_id)
        records = []
        counterexample = None
        failures = 0
        invalid_input_checks = []
        for index, config in enumerate(plans()):
            for poison in ("finite", "nan", "infinity"):
                seed = 713 + index
                values = inputs(config, poison, seed)
                assert independently_legal(values), (index, poison)
                candidate.validate_inputs(*values, require_cuda=False)
                target = independent_reference(values)
                public_target = candidate.reference(*values).numpy()
                reference_gap = float(np.max(np.abs(public_target - target)))
                assert reference_gap < 1e-11 and np.isfinite(target).all(), reference_gap
                device_values = tuple(t.cuda() for t in values[:-1]) + (values[-1],)
                snapshots = [t.clone() for t in device_values[:-1]]
                output = candidate.run(*device_values)
                torch.cuda.synchronize()
                array = output.cpu().numpy().astype(np.float64)
                finite = bool(np.isfinite(array).all())
                ratio = float(np.max(np.abs(array - target) / (0.003 + 0.003 * np.abs(target)))) if finite else None
                preserved = all(torch.equal(t.view(torch.uint8), before.view(torch.uint8))
                                for t, before in zip(device_values[:-1], snapshots))
                output_contract = (output.shape == device_values[0].shape
                                   and output.dtype == torch.float16 and output.is_contiguous()
                                   and output.device == device_values[0].device)
                passed = finite and ratio <= 1.0 and preserved and output_contract
                record = {"plan": index, "config": config, "inactive_slots": poison,
                          "seed": seed, "legal_input": True, "passed": passed,
                          "finite": finite, "max_error_ratio": ratio,
                          "inputs_preserved": preserved, "output_contract": output_contract,
                          "public_reference_max_abs_difference": reference_gap}
                if not passed:
                    failures += 1
                    difference = ~np.isfinite(array) | (np.abs(array - target) > (0.003 + 0.003 * np.abs(target)))
                    first = tuple(int(x) for x in np.argwhere(difference)[0])
                    value = float(array[first])
                    record["first_mismatch"] = {"index": first, "actual": value if np.isfinite(value) else str(value),
                                                "expected": float(target[first])}
                    if counterexample is None:
                        counterexample = dict(record)
                records.append(record)

        # This is a declared inactive-slot domain, not permission for NaN queries
        # or live KV data. Assert that the public validator rejects those cases.
        for name in ("q", "k", "v"):
            values = list(inputs(plans()[0], "nan", 713))
            if name == "q":
                values[0][0, 0, 0, 0] = float("nan")
            else:
                logical = int(values[4][0]) - 1
                physical = int(values[3][0, logical // values[1].shape[1]])
                cache = values[1 if name == "k" else 2]
                cache[physical, logical % cache.shape[1], 0, 0] = float("nan")
            try:
                candidate.validate_inputs(*values, require_cuda=False)
                rejected = False
            except AssertionError:
                rejected = True
            assert rejected, f"{case_id}: accepted illegal NaN in {name}"
            invalid_input_checks.append({"nan_location": name, "rejected": rejected})
        qualified = failures > 0 if truth == "reject" else failures == 0
        cases[case_id] = {"ground_truth": truth, "qualification_passed": qualified,
                          "total_runs": len(records), "failing_runs": failures,
                          "counterexample": counterexample, "tests": records,
                          "invalid_input_checks": invalid_input_checks,
                          "label_scope": "entire shared public contract, not per-input or seed",
                          "evidence_limitation": "Finite qualification plus indexing/masking/numerical source audit supports acceptance; successful sampling is not a formal all-input proof."}
    return {"family": "paged_attention_pair", "expected_labels": EXPECTED,
            "cases": cases, "qualification_passed": all(r["qualification_passed"] for r in cases.values())}
