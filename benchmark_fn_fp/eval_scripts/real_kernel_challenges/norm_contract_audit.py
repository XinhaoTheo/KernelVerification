"""Additive contract audit; never modifies original cases or their records."""
from pathlib import Path
import importlib.util
import hashlib
import sys

REASON = (
    "The original contract did not exclude contiguous lazy-negative PyTorch "
    "tensors. Raw pointers ignore that logical flag. Initial case_108 trust "
    "qualification omitted this legal input class; old evidence remains intact. "
    "A concrete legal counterexample corrects the label to reject."
)


def _base():
    path = Path(__file__).with_name("norm_pair.py")
    spec = importlib.util.spec_from_file_location("norm_contract_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def logical_inputs(rows=513, cols=31):
    import torch
    x = -torch.ones((rows, cols), dtype=torch.float16)
    w = torch.linspace(.5, 1.5, cols).half()
    dy = ((torch.arange(cols) % 5).float() / 8. + .25)[None, :].expand(rows, cols).contiguous().half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    return x, w, dy, rstd


def with_flags(values, names):
    import torch
    labels = ("x", "weight", "dy", "rstd")
    return tuple(torch._neg_view(-value) if name in names else value
                 for name,value in zip(labels,values))


def cpu_audit():
    import torch
    base = _base()
    plan = [("x",), ("weight",), ("dy",), ("rstd",), ("x","weight","dy","rstd")]
    results = []
    for flags in plan:
        values = with_flags(logical_inputs(), flags)
        before = tuple(t.clone() for t in values)
        resolved = tuple(t.resolve_neg() for t in values)
        assert all(t.is_contiguous() for t in values)
        assert all(t.is_contiguous() and not t.is_neg() for t in resolved)
        assert all(torch.equal(a,b) for a,b in zip(before,values))
        assert all(torch.equal(a,b) for a,b in zip(resolved,values))
        assert base._legal(resolved)
        results.append({"flags":list(flags),"legal":True,"inputs_preserved":True,
                        "resolved_contiguous":True,"logical_values_preserved":True})
    return results


def validate(case_root: Path) -> dict:
    import numpy as np
    import torch
    import triton
    base = _base()
    # A fresh in-memory helper instance: immutable historical helper file stays as-is.
    base.EXPECTED = {"case_111": "trust"}
    ordinary = base.validate(case_root)
    report = {"family":"norm_contract_audit", "cases":{},
              "environment":ordinary["environment"], "contract_unchanged":True,
              "label_correction_reason":REASON, "cpu_resolve_neg_audit":cpu_audit()}
    plans = [(rows,cols,flags) for rows,cols in ((31,31),(513,127))
             for flags in (("x",),("weight",),("dy",),("rstd",),("x","weight","dy","rstd"))]
    for case, expected in (("case_108","reject"),("case_111","trust")):
        module = base._load(case_root / case / "kernel.py")
        record = ordinary["cases"][case] if case == "case_111" else {
            "expected_label":expected,"tests":[],
            "kernel_sha256":hashlib.sha256((case_root/case/"kernel.py").read_bytes()).hexdigest()}
        for rows, cols, flags in plans:
            host = logical_inputs(rows,cols)
            assert base._legal(host)
            refs, limits = base.independent_reference(host)
            gpu = with_flags(tuple(t.cuda() for t in host), flags)
            assert all(t.is_contiguous() for t in gpu)
            # Independent oracle consumes logical values, not underlying storage.
            before = tuple(t.clone() for t in gpu)
            public = module.reference(*gpu)
            public_error = max(float(np.max(np.abs(t.double().cpu().numpy()-ref)))
                               for t,ref in zip(public,refs))
            assert public_error < 1e-8, (case,flags,public_error)
            ratios, abs_errors = [0.,0.], [0.,0.]
            unchanged, finite, shape_dtype = True, True, True
            first_mismatch = None
            for _ in range(3):
                outputs = module.run(*gpu)
                torch.cuda.synchronize()
                unchanged = unchanged and all(torch.equal(a,b) for a,b in zip(gpu,before))
                shape_dtype = shape_dtype and outputs[0].shape == (rows,cols) and outputs[1].shape == (cols,)
                shape_dtype = shape_dtype and outputs[0].dtype == torch.float16 and outputs[1].dtype == torch.float32
                for i,(output,ref,limit) in enumerate(zip(outputs,refs,limits)):
                    arr = output.double().cpu().numpy()
                    errors = np.abs(arr-ref)
                    finite = finite and bool(np.isfinite(arr).all())
                    ratios[i] = max(ratios[i],float(np.max(errors/limit)))
                    abs_errors[i] = max(abs_errors[i],float(np.max(errors)))
                    if first_mismatch is None and np.any(errors > limit):
                        index = np.unravel_index(int(np.argmax(errors/limit)),errors.shape)
                        first_mismatch = {"output":("dx","dw")[i],"index":[int(j) for j in index],
                                          "expected":float(ref[index]),"actual":float(arr[index]),
                                          "allowed_error":float(limit[index])}
            passed = finite and unchanged and shape_dtype and max(ratios)<=1
            record["tests"].append({"shape":[rows,cols],"style":"lazy_negative_view",
                "flagged_inputs":list(flags),"logical_x":-1.0,"legal":True,"repetitions":3,
                "passed":bool(passed),"max_error_ratio":dict(zip(("dx","dw"),ratios)),
                "max_abs_error":dict(zip(("dx","dw"),abs_errors)),"first_mismatch":first_mismatch,
                "inputs_preserved":unchanged,"finite":finite,"shape_dtype_valid":shape_dtype,
                "public_reference_max_difference":public_error})
        failures = [t for t in record["tests"] if not t["passed"]]
        legal_flag_failures = [t for t in failures if t.get("style") == "lazy_negative_view"]
        record["tests_total"] = len(record["tests"])
        record["tests_passed"] = len(record["tests"])-len(failures)
        record["counterexamples"] = failures
        record["qualification_passed"] = (not failures if expected=="trust" else bool(legal_flag_failures))
        record["qualified"] = record["qualification_passed"]
        record["ground_truth"] = expected if record["qualification_passed"] else None
        if case == "case_108":
            record["previous_ground_truth"] = "trust"
            record["label_correction_reason"] = REASON
        report["cases"][case] = record
    report["qualification_passed"] = all(v["qualification_passed"] for v in report["cases"].values())
    return report
