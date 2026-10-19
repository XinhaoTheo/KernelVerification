# Agentic Verification Transcript

- Entry: `case_125`
- Turns: 7
- Tool events: 10
- Claims: 1
- Description updates: 0
- LLM time: 109.274s across 6 calls | tokens: in=107276 out=6426 total=113702
- Stop reason: `verdict_recorded`
- Verdict: `trust`
- Confidence: 0.95

## Timeline

### Turn 1 - `orchestrator`

Message:

```text
Loading artifact context for the agent.
```

Tool calls:

- `load_artifact -> t1 ok`
  Args:
  {
    "entry": "case_125"
  }
  Output summary:
  {
    "entry": "case_125"
  }
- `inspect_problem -> t2 ok`
  Args:
  {
    "entry": "case_125"
  }
  Output summary:
  {
    "entry": "case_125"
  }
- `list_artifact_files -> t3 ok`
  Args:
  {
    "entry": "case_125"
  }
  Output summary:
  {
    "entry": "case_125"
  }
- `read_artifact_file -> t4 ok`
  Args:
  {
    "entry": "case_125",
    "path": "meta.json"
  }
  Output summary:
  {
    "entry": "case_125"
  }

### Turn 2 - `solo` (41.091532254s, in=11626 out=2413)

Message:

```text
The kernel implements the gemmlowp double-rounding arithmetic via 64-bit limb manipulation. Rather than proving the limb algebra analytically, I'll record a claim that the fused Triton arithmetic may deviate from the contract's exact two-stage definition (especially tie handling and extreme values), then test against `reference()` on adversarial inputs including exact-tie cases.
```

Tool calls:

- `record_claim -> t5 ok`
  Args:
  {
    "rationale": "The kernel fuses H (round-half-up at 2^31) and R (round-half-away-from-zero at 2^r) into one limb computation with sign-dependent offsets; the fused arithmetic, its int32 overflow behavior at extreme accumulator values, or the R-stage tie rule may deviate from the exact contract definition.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Contract defines exact integer arithmetic: H = floor((P+2^30)/2^31), R = nearest to H/2^r with exact half ties AWAY from zero, Q = clamp, target = (Q-z)*s, absolute error <=1e-6 for the ENTIRE legal domain including every int32 accumulator value."
      }
    ],
    "scope_rationale": "The contract requires the exact double-rounding target for every legal int32 accumulator value, per-channel
  ...[truncated 382 chars]
  Output summary:
  {
    "id": "c1",
    "rationale": "The kernel fuses H (round-half-up at 2^31) and R (round-half-away-from-zero at 2^r) into one limb computation with sign-dependent offsets; the fused arithmetic, its int32 overflow behavior at extreme accumulator values, or the R-stage tie rule may deviate from the exact contract definition.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Contract defines exact integer arithmetic: H = floor((P+2^30)/2^31), R = nearest to H/2^r with exact half ties AWAY from zero, Q = clamp, target = (Q-z)*s, absolute error <=1e-6 for the ENTIRE legal domain including every int32 accumulator value."
      }
    ],
    "scope_rationale": "The contract requires the exact double-rounding target for every legal int32 accumulator value, per-channel multiplier in [2^30,2^31-1], shift in [0,31], zero point in [-128,127], and power-of-two scale; any elementwise mismatch >1e-6 on such inputs violates the contract.",
    "statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
    "s
  ...[truncated 16 chars]

### Turn 3 - `solo` (24.048128060000003s, in=13174 out=1438)

Message:

```text
I'll run a decisive probe comparing `run()` against the contract's `reference()` on adversarial inputs: extreme accumulators, all shift values 0–31, both tie directions at the R stage, and saturation-boundary cases.
```

Tool calls:

- `run_claim_probe -> t6 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import sys, torch, importlib.util\nspec = importlib.util.spec_from_file_location(\"k\", \"/root/cases/case_125/kernel.py\")\nk = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)\n\ndev = \"cuda\"\n# Channels: one per shift value 0..31, plus a couple of extras\nshifts = torch.arange(0, 32, dtype=torch.int32)\nC = 32\nmult = torch.full((C,), 2**31 - 1, dtype=torch.int32)\n# vary multipliers across channels\ng = torch.Generator(\"cpu\").manual_seed(1)\nmult = torch.randint(2**30, 2**31, (C,), dtype=torch.int64, generator=g).to(torch.int32)\nzeros = torch.tensor([0,127,-128] * 11, dtype=torch.int32)[:C]\nexps = torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1,0] * 4, dtype=torch.int32)[:C]\nscales = torch.pow(torch.tensor(2.0), exps.to(torch.float64)).to(
  ...[truncated 2933 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t6_probe.py",
        "sha256": "1a0e5430e510a9c0364e839efce2f7c5b5b92f76ba5162f5af18a433bb842b3c"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t6_stdout.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t6_stderr.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
    "duration_s": 60.03762,
    "event_id": "t6",
    "evidence_draft": {
      "claim_id": "c1",
      "data": {
        "artifacts": [
          {
            "description": "Python probe code executed by local runtime.",
            "kind": "probe_code",
            "path": "probe
  ...[truncated 1275 chars]

### Turn 4 - `solo` (15.197277966999991s, in=16162 out=917)

Message:

```text
The first probe timed out during the slow Python-side tie search, so it produced no evidence. I'll rerun a leaner, vectorized version with a longer timeout.
```

Tool calls:

- `run_claim_probe -> t7 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, importlib.util, json\nspec = importlib.util.spec_from_file_location(\"k\", \"/root/cases/case_125/kernel.py\")\nk = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)\ndev=\"cuda\"\nC=32\nshifts=torch.arange(0,32,dtype=torch.int32)\ng=torch.Generator(\"cpu\").manual_seed(1)\nmult=torch.randint(2**30,2**31,(C,),dtype=torch.int64,generator=g).to(torch.int32)\nzeros=torch.tensor([0,127,-128]*11,dtype=torch.int32)[:C]\nexps=torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1,0]*4,dtype=torch.int32)[:C]\nscales=torch.pow(torch.tensor(2.0),exps.to(torch.float64)).to(torch.float32)\nm64=mult.to(torch.int64)\n\n# Big pool of candidate accumulators (vectorized) \u2014 300k random int32\npool=torch.randint(-(2**31),2**31,(300000,C),dtype=torch.int64,ge
  ...[truncated 1799 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t7_probe.py",
        "sha256": "33670d35daa04fbcd928be1024776b2b750bd3105bf2ab9f0a7c90c4ae142511"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t7_stdout.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t7_stderr.txt",
        "sha256": "e633f54b0c046fe8b00c606fcac9daa92238155ee57f09eac6e105ddd3365bce"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
    "duration_s": 3.215512,
    "event_id": "t7",
    "evidence_draft": {
      "claim_id": "c1",
      "data": {
        "artifacts": [
          {
            "description": "Python probe code executed by local runtime.",
            "kind": "probe_code",
            "path": "probe
  ...[truncated 1876 chars]

### Turn 5 - `solo` (13.718979268000027s, in=19021 out=925)

Message:

```text
The previous probe had an indexing bug in my tie-search code. I'll fix it and rerun.
```

Tool calls:

- `run_claim_probe -> t8 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, importlib.util, json\nspec = importlib.util.spec_from_file_location(\"k\", \"/root/cases/case_125/kernel.py\")\nk = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)\ndev=\"cuda\"; C=32\nshifts=torch.arange(0,32,dtype=torch.int32)\ng=torch.Generator(\"cpu\").manual_seed(1)\nmult=torch.randint(2**30,2**31,(C,),dtype=torch.int64,generator=g).to(torch.int32)\nzeros=torch.tensor([0,127,-128]*11,dtype=torch.int32)[:C]\nexps=torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1,0]*4,dtype=torch.int32)[:C]\nscales=torch.pow(torch.tensor(2.0),exps.to(torch.float64)).to(torch.float32)\nm64=mult.to(torch.int64)\n\npool=torch.randint(-(2**31),2**31,(400000,C),dtype=torch.int64,generator=torch.Generator(\"cpu\").manual_seed(3))\nP=pool*m64\nH=(P+(1<<30))/
  ...[truncated 1887 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t8_probe.py",
        "sha256": "d5be32d5d47612fa0824158eb51110e28b712c4b567c4725d781a7586320adaf"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t8_stdout.txt",
        "sha256": "dba27150f0256448c4280c84704421a4ccadf5bcc88a1b92b4d108a75bfa6e00"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t8_stderr.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Parsed JSON object from the last non-empty stdout line.",
        "kind": "json_result",
        "path": "probes/t8_json_result.json",
        "sha256": "a8fec30805d5b5c8baa2642ec964cfecbfcfe769f281e1eacf59cdd7b1f2133d"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
    "duration_s":
  ...[truncated 2101 chars]

### Turn 6 - `solo` (8.158639869000012s, in=22020 out=415)

Message:

```text
The t8 probe passed with zero mismatches on adversarial inputs (extreme accumulators ±2^31, exact R-stage ties both signs, H-stage ties, saturation boundaries, and 200 random rows across all 32 shifts). I'll record this as rebutting evidence, resolve c1, and issue the verdict.
```

Tool calls:

- `finalize_probe_evidence -> t9 ok`
  Args:
  {
    "data": {
      "adversarial_coverage": [
        "\u00b12^31",
        "0",
        "R-stage exact ties +/-",
        "H-stage exact ties",
        "saturation boundary H 127/128/129/-127/-128/-129/255/256/-255/-256",
        "200 random rows"
      ],
      "max_abs_err": 0.0,
      "mismatches": 0,
      "shape": [
        276,
        32
      ]
    },
    "event_id": "t8",
    "status": "rebutted",
    "summary": "Kernel run() vs contract reference() on [276,32] input covering all shift values 0-31, random multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 scales, accumulators including -2^31, 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values, and 200 random int32 rows: max abs error 0.0, 0 mismatches >1e-6.",
    "supports": "rebutted"
  }
  Output summary:
  {
    "claim_id": "c1",
    "claim_status": "rebutted",
    "evidence_id": "c1.e1",
    "summary": "Kernel run() vs contract reference() on [276,32] input covering all shift values 0-31, random multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 scales, accumulators including -2^31, 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values, and 200 random int32 rows: max abs error 0.0, 0 mismatches >1e-6.",
    "supports": "rebutted",
    "tool_event_id": "t8"
  }

### Turn 7 - `solo` (7.059119409999994s, in=25273 out=318)

Message:

```text
The sole correctness claim has been resolved: the adversarial probe (event t8) showed zero mismatches and exact agreement with the contract reference across extremes, ties, and saturation boundaries. I'll record the verdict.
```

Tool calls:

- `record_verdict -> t10 ok`
  Args:
  {
    "confidence": 0.95,
    "decisive_claims": [
      "c1"
    ],
    "reason": "The kernel implements the contract's gemmlowp double-rounding epilogue (H = floor((P+2^30)/2^31), R = nearest with half ties away from zero, saturation after zero-point addition, output (Q-z)*s). Probe t8 compared run() against the kernel's exact Python arbitrary-precision reference() on a [276,32] input covering all 32 shift values, multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 power-of-two scales, accumulators \u00b12^31 and 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values (\u00b1127/128/129/255/256), and 200 random int32 rows: max absolute error 0.0 and 0 mismatches >1e-6. Since the fused limb arithmetic reproduces the exact two-stage definiti
  ...[truncated 118 chars]
  Output summary:
  {
    "confidence": 0.95,
    "decisive_claims": [
      "c1"
    ],
    "reason": "The kernel implements the contract's gemmlowp double-rounding epilogue (H = floor((P+2^30)/2^31), R = nearest with half ties away from zero, saturation after zero-point addition, output (Q-z)*s). Probe t8 compared run() against the kernel's exact Python arbitrary-precision reference() on a [276,32] input covering all 32 shift values, multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 power-of-two scales, accumulators \u00b12^31 and 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values (\u00b1127/128/129/255/256), and 200 random int32 rows: max absolute error 0.0 and 0 mismatches >1e-6. Since the fused limb arithmetic reproduces the exact two-stage definition even at tie and overflow-prone extremes, the contract is satisfied; claim c1 was rebutted.",
    "verdict": "trust"
  }

## Description Model

No description model recorded.

## Claims

### c1 - `rebutted`

Statement: The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.

Scope: `in_scope`

Scope rationale: The contract requires the exact double-rounding target for every legal int32 accumulator value, per-channel multiplier in [2^30,2^31-1], shift in [0,31], zero point in [-128,127], and power-of-two scale; any elementwise mismatch >1e-6 on such inputs violates the contract.

Scope evidence:
- `problem.txt`: Contract defines exact integer arithmetic: H = floor((P+2^30)/2^31), R = nearest to H/2^r with exact half ties AWAY from zero, Q = clamp, target = (Q-z)*s, absolute error <=1e-6 for the ENTIRE legal domain including every int32 accumulator value.

Rationale: The kernel fuses H (round-half-up at 2^31) and R (round-half-away-from-zero at 2^r) into one limb computation with sign-dependent offsets; the fused arithmetic, its int32 overflow behavior at extreme accumulator values, or the R-stage tie rule may deviate from the exact contract definition.

Evidence:
- `c1.e1` runtime_probe supports `rebutted`, tool_event_id=t8: Kernel run() vs contract reference() on [276,32] input covering all shift values 0-31, random multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 scales, accumulators including -2^31, 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values, and 200 random int32 rows: max abs error 0.0, 0 mismatches >1e-6.

## Tool Events

### t1 - load_artifact - ok

{
  "entry": "case_125"
}

### t2 - inspect_problem - ok

{
  "entry": "case_125"
}

### t3 - list_artifact_files - ok

{
  "entry": "case_125"
}

### t4 - read_artifact_file - ok

{
  "entry": "case_125"
}

### t5 - record_claim - ok

{
  "id": "c1",
  "rationale": "The kernel fuses H (round-half-up at 2^31) and R (round-half-away-from-zero at 2^r) into one limb computation with sign-dependent offsets; the fused arithmetic, its int32 overflow behavior at extreme accumulator values, or the R-stage tie rule may deviate from the exact contract definition.",
  "scope": "in_scope",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "Contract defines exact integer arithmetic: H = floor((P+2^30)/2^31), R = nearest to H/2^r with exact half ties AWAY from zero, Q = clamp, target = (Q-z)*s, absolute error <=1e-6 for the ENTIRE legal domain including every int32 accumulator value."
    }
  ],
  "scope_rationale": "The contract requires the exact double-rounding target for every legal int32 accumulator value, per-channel multiplier in [2^30,2^31-1], shift in [0,31], zero point in [-128,127], and power-of-two scale; any elementwise mismatch >1e-6 on such inputs violates the contract.",
  "statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
  "status": "open"
}

### t6 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t6_probe.py",
      "sha256": "1a0e5430e510a9c0364e839efce2f7c5b5b92f76ba5162f5af18a433bb842b3c"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t6_stdout.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t6_stderr.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
  "duration_s": 60.03762,
  "event_id": "t6",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t6_probe.py",
          "sha256": "1a0e5430e510a9c0364e839efce2f7c5b5b92f76ba5162f5af18a433bb842b3c"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t6_stdout.txt",
          "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t6_stderr.txt",
          "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        }
      ],
      "duration_s": 60.03762,
      "exit_code": null,
      "expected_signal": "mismatches == 0 (or >0 with details)",
      "json_parse_error": "stdout has no non-empty lines",
      "
...[truncated 475 chars]

### t7 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t7_probe.py",
      "sha256": "33670d35daa04fbcd928be1024776b2b750bd3105bf2ab9f0a7c90c4ae142511"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t7_stdout.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t7_stderr.txt",
      "sha256": "e633f54b0c046fe8b00c606fcac9daa92238155ee57f09eac6e105ddd3365bce"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
  "duration_s": 3.215512,
  "event_id": "t7",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t7_probe.py",
          "sha256": "33670d35daa04fbcd928be1024776b2b750bd3105bf2ab9f0a7c90c4ae142511"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t7_stdout.txt",
          "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t7_stderr.txt",
          "sha256": "e633f54b0c046fe8b00c606fcac9daa92238155ee57f09eac6e105ddd3365bce"
        }
      ],
      "duration_s": 3.215512,
      "exit_code": 1,
      "expected_signal": "mismatches == 0 (or >0 with details)",
      "json_parse_error": "stdout has no non-empty lines",
      "jso
...[truncated 1076 chars]

### t8 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t8_probe.py",
      "sha256": "d5be32d5d47612fa0824158eb51110e28b712c4b567c4725d781a7586320adaf"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t8_stdout.txt",
      "sha256": "dba27150f0256448c4280c84704421a4ccadf5bcc88a1b92b4d108a75bfa6e00"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t8_stderr.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Parsed JSON object from the last non-empty stdout line.",
      "kind": "json_result",
      "path": "probes/t8_json_result.json",
      "sha256": "a8fec30805d5b5c8baa2642ec964cfecbfcfe769f281e1eacf59cdd7b1f2133d"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "The kernel's fused fixed-point computation deviates from the contract's exact two-stage rounding (H floor, R half-away-from-zero) on some legal input, producing elementwise absolute error > 1e-6.",
  "duration_s": 7.05619,
  "event_id": "t8",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t8_probe.py",
          "sha256": "d5be32d5d47612fa0824158eb51110e28b712c4b567c4725d781a7586320adaf"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t8_stdout.txt",
          "sha256": "dba27150f0256448c4280c84704421a4ccadf5bcc88a1b92b4d108a75bfa6e00"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t8_stderr.txt",
          "sha256": "e3b0c44298fc1c149afbf4c
...[truncated 1301 chars]

### t9 - finalize_probe_evidence - ok

{
  "claim_id": "c1",
  "claim_status": "rebutted",
  "evidence_id": "c1.e1",
  "summary": "Kernel run() vs contract reference() on [276,32] input covering all shift values 0-31, random multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 scales, accumulators including -2^31, 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values, and 200 random int32 rows: max abs error 0.0, 0 mismatches >1e-6.",
  "supports": "rebutted",
  "tool_event_id": "t8"
}

### t10 - record_verdict - ok

{
  "confidence": 0.95,
  "decisive_claims": [
    "c1"
  ],
  "reason": "The kernel implements the contract's gemmlowp double-rounding epilogue (H = floor((P+2^30)/2^31), R = nearest with half ties away from zero, saturation after zero-point addition, output (Q-z)*s). Probe t8 compared run() against the kernel's exact Python arbitrary-precision reference() on a [276,32] input covering all 32 shift values, multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 power-of-two scales, accumulators \u00b12^31 and 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values (\u00b1127/128/129/255/256), and 200 random int32 rows: max absolute error 0.0 and 0 mismatches >1e-6. Since the fused limb arithmetic reproduces the exact two-stage definition even at tie and overflow-prone extremes, the contract is satisfied; claim c1 was rebutted.",
  "verdict": "trust"
}

## Verdict

{
  "confidence": 0.95,
  "created_at": "2026-10-05T06:57:52Z",
  "decisive_claims": [
    "c1"
  ],
  "reason": "The kernel implements the contract's gemmlowp double-rounding epilogue (H = floor((P+2^30)/2^31), R = nearest with half ties away from zero, saturation after zero-point addition, output (Q-z)*s). Probe t8 compared run() against the kernel's exact Python arbitrary-precision reference() on a [276,32] input covering all 32 shift values, multipliers in [2^30,2^31), zero points {0,127,-128}, all 9 power-of-two scales, accumulators \u00b12^31 and 2^31-1, exact R-stage half ties (both signs per channel), exact H-stage ties, saturation-boundary H values (\u00b1127/128/129/255/256), and 200 random int32 rows: max absolute error 0.0 and 0 mismatches >1e-6. Since the fused limb arithmetic reproduces the exact two-stage definition even at tie and overflow-prone extremes, the contract is satisfied; claim c1 was rebutted.",
  "verdict": "trust"
}
