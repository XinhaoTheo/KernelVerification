"""Independent T4 qualification; never invokes an evaluation model.

Labels are declared by each construction before execution. Tests qualify or
disqualify that declaration; a lucky input never changes a kernel's label.
Only this private oracle container receives the qualification helpers.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import modal

BENCHMARK = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/root/oracles")
PRIVATE = BENCHMARK / "real_kernel_challenges/private_data"
FAMILIES = {"state_pair": ("case_33", "case_34"),
            "norm_pair": ("case_35", "case_36"),
            "state_contract_audit": ("case_34", "case_37"),
            "norm_contract_audit": ("case_35", "case_38"),
            "segmented_state_pair": ("case_39", "case_40"),
            "paged_attention_pair": ("case_41", "case_42"),
            "splitk_pair": ("case_43", "case_44"),
            "moe_pair": ("case_45", "case_46"),
            "int4_pair": ("case_47", "case_48"),
            "stencil_pair": ("case_49", "case_50"),
            "requant_pair": ("case_51", "case_52"),
            "erf_pair": ("case_53", "case_54"),
            "gelu_pair": ("case_55", "case_56"),
            "silu_pair": ("case_57", "case_58"),
            "sine_pair": ("case_59", "case_60"),
            "hypot_pair": ("case_61", "case_62"),
            "softplus_pair": ("case_63", "case_64"),
            "compensated_pair": ("case_65", "case_66"),
            "fastdiv_pair": ("case_67", "case_68"),
            "butterfly_pair": ("case_69", "case_70"),
            "packed_argmax_pair": ("case_71", "case_72"),
            "atan_pair": ("case_73", "case_74"),
            "rounding_pair": ("case_75", "case_76"),
            "ntt_pair": ("case_77", "case_78"),
            "crc_pair": ("case_79", "case_80"),
            "complex_pair": ("case_81", "case_82"),
            "requant_compose_pair": ("case_83", "case_84"),
            "affine_scan_pair": ("case_85", "case_86"),
            "winograd_pair": ("case_87", "case_88"),
            "realfft_pair": ("case_89", "case_90"),
            "exp2_pair": ("case_163", "case_164"),
            "log1p_pair": ("case_165", "case_166"),
            "aes_pair": ("case_167", "case_168")}
app = modal.App("kv-real-kernel-qualification")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4",
                      "gmpy2==2.2.1", "mpmath==1.3.0"))
if modal.is_local():
    image = (image.add_local_dir(str(BENCHMARK / "triton_eval_cases"), "/root/cases")
             .add_local_dir(str(HERE), "/root/oracles"))


@app.function(image=image, gpu="T4", timeout=1800)
def qualify(family_names: list[str]) -> dict:
    import platform
    import traceback
    import torch
    import triton
    import numpy as np

    sys.path.insert(0, "/root/oracles")
    cases, families = {}, {}
    for family_name in family_names:
        source = Path("/root/oracles") / f"{family_name}.py"
        try:
            spec = importlib.util.spec_from_file_location(family_name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[family_name] = module
            spec.loader.exec_module(module)
            result = module.validate(Path("/root/cases"))
            families[family_name] = result
            for name, row in result["cases"].items():
                if name not in FAMILIES[family_name] or name in cases:
                    raise ValueError(f"Unexpected or duplicate case: {name}")
                row = dict(row)
                for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
                    row[f"{kind}_sha256"] = hashlib.sha256(
                        (Path("/root/cases") / name / filename).read_bytes()).hexdigest()
                row["oracle_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                row["family"] = family_name
                cases[name] = row
        except Exception:
            families[family_name] = {"qualification_passed": False,
                                     "error": traceback.format_exc()}
    return {"environment": {"gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "triton": triton.__version__,
            "numpy": np.__version__, "python": platform.python_version()},
            "families": families, "cases": cases}


@app.local_entrypoint()
def main(families: str = "state_pair,norm_pair"):
    # Frozen evidence keeps its original IDs. Directory numbering is resolved
    # only at this local read/write boundary, never by rewriting saved evidence.
    sys.path.insert(0, str(BENCHMARK / "eval_scripts"))
    from case_registry import load_registry

    registry = load_registry(BENCHMARK)
    details = registry.get("case_details", {})
    names = [name.strip() for name in families.split(",") if name.strip()]
    if not names or len(set(names)) != len(names) or any(name not in FAMILIES for name in names):
        raise ValueError("Select supported, unique families")
    expected_cases = {case for name in names for case in FAMILIES[name]}
    before = {name: {f"{kind}_sha256": hashlib.sha256(
        (BENCHMARK / "triton_eval_cases" / name / filename).read_bytes()).hexdigest()
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}
        for name in expected_cases}
    result = qualify.remote(names)
    attempts = PRIVATE / "qualification"
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f"r{number}.json").exists():
        number += 1
    destination = attempts / f"r{number}.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Qualification evidence: {destination}", flush=True)
    for name, row in result["cases"].items():
        print(name, row.get("ground_truth"), "qualified=", row.get("qualification_passed"), flush=True)
    for family, row in result["families"].items():
        if row.get("error"):
            print(family, row["error"], flush=True)
    if set(result["cases"]) != expected_cases:
        raise RuntimeError("Missing case qualification; full result retained")
    for name, row in result["cases"].items():
        if not row.get("qualification_passed") or row.get("ground_truth") not in {"trust", "reject"}:
            raise RuntimeError(f"Case is not independently qualified: {name}")
        for key, value in before[name].items():
            if row.get(key) != value:
                raise RuntimeError(f"Mounted source changed: {name}/{key}")
    frozen_path = PRIVATE / "validation_gpu.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {"cases": {}}
    for name, row in result["cases"].items():
        detail = details.get(name, {})
        frozen_id = detail.get("validation_case_id", name)
        owners = [case for case, entry in details.items()
                  if entry.get("dataset") == "real_kernel_challenges"
                  and entry.get("validation_case_id", case) == frozen_id]
        if owners and owners != [name]:
            raise RuntimeError(f"Ambiguous frozen validation ID: {name}/{frozen_id}")
        prior = frozen["cases"].get(frozen_id)
        if prior and any(prior.get(k) != v for k, v in before[name].items()):
            raise RuntimeError(f"Previously frozen case edited: {name}")
        if prior and prior["ground_truth"] != row["ground_truth"] and not row.get("label_correction_reason"):
            raise RuntimeError(f"Label correction must have explicit evidence and reason: {name}")
        frozen["cases"][frozen_id] = {key: row[key] for key in
            ("ground_truth", "qualification_passed", "kernel_sha256", "problem_sha256",
             "oracle_source_sha256", "family")}
        frozen["cases"][frozen_id].update(environment=result["environment"],
            evidence_file=destination.relative_to(PRIVATE).as_posix(),
            evidence_case_id=name,
            label_scope="fixed implementation under the entire public contract; not per-input compliance")
        if prior and prior["ground_truth"] != row["ground_truth"]:
            frozen["cases"][frozen_id]["label_correction"] = {
                "previous_ground_truth": prior["ground_truth"],
                "previous_evidence_file": prior["evidence_file"],
                "reason": row["label_correction_reason"],
                "source_and_contract_unchanged": True,
                "historical_traces_unchanged": True,
            }
    frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    print(f"Frozen {len(expected_cases)} cases: {frozen_path}", flush=True)
