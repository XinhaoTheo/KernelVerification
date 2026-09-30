# Agentic Verification Transcript

- Entry: `case_07`
- Turns: 7
- Tool events: 10
- Claims: 1
- Description updates: 0
- LLM time: 138.236s across 6 calls | tokens: in=77901 out=4683 total=82584
- Stop reason: `verdict_recorded`
- Verdict: `trust`
- Confidence: 0.85

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
    "entry": "case_07"
  }
  Output summary:
  {
    "entry": "case_07"
  }
- `inspect_problem -> t2 ok`
  Args:
  {
    "entry": "case_07"
  }
  Output summary:
  {
    "entry": "case_07"
  }
- `list_artifact_files -> t3 ok`
  Args:
  {
    "entry": "case_07"
  }
  Output summary:
  {
    "entry": "case_07"
  }
- `read_artifact_file -> t4 ok`
  Args:
  {
    "entry": "case_07",
    "path": "meta.json"
  }
  Output summary:
  {
    "entry": "case_07"
  }

### Turn 2 - `solo` (24.704965111000007s, in=8432 out=1332)

Message:

```text
This kernel appears to be a near-copy of the original AutoGPTQ `quant_matmul_248` kernel. I'll verify it numerically against a reference dequantize+matmul on representative shapes.
```

Tool calls:

- `record_claim -> t5 error`
  Args:
  {
    "claim_id</arg_value><arg_value>c1</arg_value><arg_key>rationale": "The kernel indexes packed b by k//8, g_idx by k, scales/zeros by group and column, and adds +1 to the zero point, matching the AutoGPTQ convention; a numerical mismatch against an independent dequant reference would indicate an indexing or packing error.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "dequant(B)[k,n] = (unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n]; kernel must evaluate C = A @ dequant(B) faithfully in the declared 4-bit format."
      }
    ],
    "scope_rationale": "The contract requires C = A @ dequant(B) with the exact dequantization formula per (group, column); any deviation in indexing/packing that changes C on inputs the contract admits i
  ...[truncated 198 chars]
  Output summary:
  {
    "error_type": "ToolRegistryError",
    "message": "tool record_claim missing required arg: rationale"
  }

### Turn 3 - `solo` (5.114870886999995s, in=9216 out=307)

Message:

```text
I'll record the claim (the previous call failed due to a malformed argument) and then run a numerical probe comparing the kernel against an independent dequantization + matmul reference.
```

Tool calls:

- `record_claim -> t6 ok`
  Args:
  {
    "rationale": "The kernel indexes packed b by k//8, g_idx by k, scales/zeros by group and column, and adds +1 to the zero point, matching the AutoGPTQ convention; a numerical mismatch against an independent dequant reference would indicate an indexing or packing error.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "dequant(B)[k,n] = (unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n]; kernel must evaluate C = A @ dequant(B) faithfully in the declared 4-bit format."
      }
    ],
    "scope_rationale": "The contract requires C = A @ dequant(B) with the exact dequantization formula per (group, column); any deviation that changes C on inputs the contract admits is a defect.",
    "statement": "gptq_matmul produces C = A @ dequant(B) matc
  ...[truncated 124 chars]
  Output summary:
  {
    "id": "c1",
    "rationale": "The kernel indexes packed b by k//8, g_idx by k, scales/zeros by group and column, and adds +1 to the zero point, matching the AutoGPTQ convention; a numerical mismatch against an independent dequant reference would indicate an indexing or packing error.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "dequant(B)[k,n] = (unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n]; kernel must evaluate C = A @ dequant(B) faithfully in the declared 4-bit format."
      }
    ],
    "scope_rationale": "The contract requires C = A @ dequant(B) with the exact dequantization formula per (group, column); any deviation that changes C on inputs the contract admits is a defect.",
    "statement": "gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.",
    "status": "open"
  }

### Turn 4 - `solo` (18.746985584000015s, in=10537 out=1089)

Message:

```text
I'll run a numerical probe comparing the kernel to an independent dequantization + matmul reference, using the AutoGPTQ packing convention (zero effective = unpacked zero + 1) on representative shapes.
```

Tool calls:

- `run_claim_probe -> t7 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "\nimport torch, json, sys\nsys.path.insert(0, '/root/cases/case_07')\nfrom kernel import gptq_matmul\n\ntorch.manual_seed(0)\ndev = 'cuda'\nM, K, N = 64, 256, 128\nbits = 4\nmaxq = 15\ngroupsize = 32\nn_groups = K // groupsize\n\na = torch.randn(M, K, device=dev, dtype=torch.float16)\nq = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)\n# pack 8 values per int32 along K (little-endian nibbles)\nb_packed = torch.zeros((K // 8, N), device=dev, dtype=torch.int32)\nfor k in range(K):\n    word, off = k // 8, k % 8\n    b_packed[word, :] |= q[k, :] << (off * bits)\nscales = (torch.randn(n_groups, N, device=dev, dtype=torch.float16).abs() * 0.01 + 0.001)\nzeros = torch.randint(0, 16, (n_groups, N), device=dev, dtype=torch.int32)  # packed zeros along 
  ...[truncated 1257 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t7_probe.py",
        "sha256": "7cb9a9f78155139da9557ec68d3f09f831b9f295646bde9c084d1af5495ba844"
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
        "sha256": "a798b590ce108d51f73d6d2ecca1a1da844e0c15bc32ef1eccad95ade3428602"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.",
    "duration_s": 5.552807,
    "event_id": "t7",
    "evidence_draft": {
      "claim_id": "c1",
      "data": {
        "artifacts": [
          {
            "description": "Python probe code executed by local runtime.",
            "kind": "probe_code",
            "path": "probes/t7_probe.py",
            "sha
  ...[truncated 2018 chars]

### Turn 5 - `solo` (20.35998751400001s, in=13307 out=1026)

Message:

```text
The probe failed due to a bug in my own reference code (zeros indexing). Note the kernel treats `zeros` as packed along N (indexed `[g, n//8]`), matching AutoGPTQ. I'll rerun with a corrected reference.
```

Tool calls:

- `run_claim_probe -> t8 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, json, sys\nsys.path.insert(0, '/root/cases/case_07')\nfrom kernel import gptq_matmul\n\ntorch.manual_seed(0)\ndev = 'cuda'\nM, K, N = 64, 256, 128\nbits, maxq, groupsize = 4, 15, 32\nn_groups = K // groupsize\n\na = torch.randn(M, K, device=dev, dtype=torch.float16)\nq = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)\nb_packed = torch.zeros((K // 8, N), device=dev, dtype=torch.int32)\nfor k in range(K):\n    b_packed[k // 8, :] |= q[k, :] << ((k % 8) * bits)\nscales = (torch.randn(n_groups, N, device=dev, dtype=torch.float16).abs() * 0.01 + 0.001)\nzeros = torch.randint(0, 16, (n_groups, N // 8), device=dev, dtype=torch.int32)  # packed along N\ng_idx = (torch.arange(K, device=dev) // groupsize).to(torch.int32)\n\nc = gptq_matmul(
  ...[truncated 904 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t8_probe.py",
        "sha256": "16a2278d6d27c9554e049f3276afbcb92928c474cd94deec1ca2aa17feb815a0"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t8_stdout.txt",
        "sha256": "987361641dfb1049fea4320904a7fa6c3f9d0ed40d2e0457e2b5b092ea0878bc"
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
        "sha256": "64915a96cabc93b009f3fc80e9a0dd1c6f75d65a5d6953c0989c8cdc0c9bc231"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.",
    "duration_s": 5.180513,
    "event_id": "t8",
  ...[truncated 2851 chars]

### Turn 6 - `solo` (13.016174847000002s, in=16180 out=615)

Message:

```text
The t8 probe succeeded: the kernel output matches an independent dequant+matmul reference with max abs error 0.0014 (fp16-accumulate rounding scale) on a representative 4-bit GPTQ input. I'll finalize that evidence and record the verdict.
```

Tool calls:

- `finalize_probe_evidence -> t9 ok`
  Args:
  {
    "data": {
      "allclose": true,
      "max_abs_err": 0.00141221284866333,
      "mean_abs_err": 0.00021785117860417813,
      "metric": "max abs error vs independent dequant reference (zero_eff = unpacked_zero + 1, AutoGPTQ convention)",
      "shape": [
        64,
        128
      ],
      "tested": "M=64, K=256, N=128, bits=4, groupsize=32, random q/scales/zeros, packed little-endian nibbles along K (b) and N (zeros)"
    },
    "event_id": "t8",
    "status": "confirmed",
    "summary": "On a representative 4-bit GPTQ input (M=64,K=256,N=128, groupsize 32, packed int32 nibbles, per-(group,column) scales and zeros), gptq_matmul matches an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1))*scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent w
  ...[truncated 210 chars]
  Output summary:
  {
    "claim_id": "c1",
    "claim_status": "confirmed",
    "evidence_id": "c1.e1",
    "summary": "On a representative 4-bit GPTQ input (M=64,K=256,N=128, groupsize 32, packed int32 nibbles, per-(group,column) scales and zeros), gptq_matmul matches an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1))*scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent with float16 matmul accumulation, i.e. faithful evaluation of the declared 4-bit format. The kernel's zeros+1 matches the AutoGPTQ packed-zero convention referenced by the contract.",
    "supports": "confirmed",
    "tool_event_id": "t8"
  }

### Turn 7 - `solo` (56.29256813499998s, in=20229 out=314)

Message:

```text
The sole claim c1 is confirmed with a successful runtime probe (t8): the kernel matches an independent dequant+matmul reference at float16-matmul rounding scale on a representative 4-bit GPTQ input. No open claims remain — I'll record the verdict.
```

Tool calls:

- `record_verdict -> t10 ok`
  Args:
  {
    "confidence": 0.85,
    "decisive_claims": [
      "c1"
    ],
    "reason": "Probe t8 (run_claim_probe, exit 0) tested gptq_matmul on a representative 4-bit GPTQ input (M=64, K=256, N=128, groupsize 32, packed int32 little-endian nibbles along K for B and along N for zeros, per-(group,column) scales) and compared against an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1)) * scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent with float16 tl.dot accumulation, i.e. faithful evaluation of the declared 4-bit format per the contract's dequantization formula. The kernel's zeros+1 handling matches the AutoGPTQ packed-zero convention referenced by problem.txt. No conflicting evidence; kernel source inspection shows correct per-group s
  ...[truncated 44 chars]
  Output summary:
  {
    "confidence": 0.85,
    "decisive_claims": [
      "c1"
    ],
    "reason": "Probe t8 (run_claim_probe, exit 0) tested gptq_matmul on a representative 4-bit GPTQ input (M=64, K=256, N=128, groupsize 32, packed int32 little-endian nibbles along K for B and along N for zeros, per-(group,column) scales) and compared against an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1)) * scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent with float16 tl.dot accumulation, i.e. faithful evaluation of the declared 4-bit format per the contract's dequantization formula. The kernel's zeros+1 handling matches the AutoGPTQ packed-zero convention referenced by problem.txt. No conflicting evidence; kernel source inspection shows correct per-group scale/zero indexing.",
    "verdict": "trust"
  }

## Description Model

No description model recorded.

## Claims

### c1 - `confirmed`

Statement: gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.

Scope: `in_scope`

Scope rationale: The contract requires C = A @ dequant(B) with the exact dequantization formula per (group, column); any deviation that changes C on inputs the contract admits is a defect.

Scope evidence:
- `problem.txt`: dequant(B)[k,n] = (unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n]; kernel must evaluate C = A @ dequant(B) faithfully in the declared 4-bit format.

Rationale: The kernel indexes packed b by k//8, g_idx by k, scales/zeros by group and column, and adds +1 to the zero point, matching the AutoGPTQ convention; a numerical mismatch against an independent dequant reference would indicate an indexing or packing error.

Evidence:
- `c1.e1` runtime_probe supports `confirmed`, tool_event_id=t8: On a representative 4-bit GPTQ input (M=64,K=256,N=128, groupsize 32, packed int32 nibbles, per-(group,column) scales and zeros), gptq_matmul matches an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1))*scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances — consistent with float16 matmul accumulation, i.e. faithful evaluation of the declared 4-bit format. The kernel's zeros+1 matches the AutoGPTQ packed-zero convention referenced by the contract.

## Tool Events

### t1 - load_artifact - ok

{
  "entry": "case_07"
}

### t2 - inspect_problem - ok

{
  "entry": "case_07"
}

### t3 - list_artifact_files - ok

{
  "entry": "case_07"
}

### t4 - read_artifact_file - ok

{
  "entry": "case_07"
}

### t5 - record_claim - error

{
  "error_type": "ToolRegistryError",
  "message": "tool record_claim missing required arg: rationale"
}

### t6 - record_claim - ok

{
  "id": "c1",
  "rationale": "The kernel indexes packed b by k//8, g_idx by k, scales/zeros by group and column, and adds +1 to the zero point, matching the AutoGPTQ convention; a numerical mismatch against an independent dequant reference would indicate an indexing or packing error.",
  "scope": "in_scope",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "dequant(B)[k,n] = (unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n]; kernel must evaluate C = A @ dequant(B) faithfully in the declared 4-bit format."
    }
  ],
  "scope_rationale": "The contract requires C = A @ dequant(B) with the exact dequantization formula per (group, column); any deviation that changes C on inputs the contract admits is a defect.",
  "statement": "gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.",
  "status": "open"
}

### t7 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t7_probe.py",
      "sha256": "7cb9a9f78155139da9557ec68d3f09f831b9f295646bde9c084d1af5495ba844"
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
      "sha256": "a798b590ce108d51f73d6d2ecca1a1da844e0c15bc32ef1eccad95ade3428602"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.",
  "duration_s": 5.552807,
  "event_id": "t7",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t7_probe.py",
          "sha256": "7cb9a9f78155139da9557ec68d3f09f831b9f295646bde9c084d1af5495ba844"
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
          "sha256": "a798b590ce108d51f73d6d2ecca1a1da844e0c15bc32ef1eccad95ade3428602"
        }
      ],
      "duration_s": 5.552807,
      "exit_code": 1,
      "expected_signal": "max abs err near float16 matmul rounding (~<0.1 for these scales)",
      "json_parse_error": "stdout has no non-empty lines",
      "json
...[truncated 1218 chars]

### t8 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t8_probe.py",
      "sha256": "16a2278d6d27c9554e049f3276afbcb92928c474cd94deec1ca2aa17feb815a0"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t8_stdout.txt",
      "sha256": "987361641dfb1049fea4320904a7fa6c3f9d0ed40d2e0457e2b5b092ea0878bc"
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
      "sha256": "64915a96cabc93b009f3fc80e9a0dd1c6f75d65a5d6953c0989c8cdc0c9bc231"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "gptq_matmul produces C = A @ dequant(B) matching the contract formula dequant(B)[k,n] = (q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n] on typical 4-bit GPTQ inputs.",
  "duration_s": 5.180513,
  "event_id": "t8",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t8_probe.py",
          "sha256": "16a2278d6d27c9554e049f3276afbcb92928c474cd94deec1ca2aa17feb815a0"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t8_stdout.txt",
          "sha256": "987361641dfb1049fea4320904a7fa6c3f9d0ed40d2e0457e2b5b092ea0878bc"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t8_stderr.txt",
          "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495
...[truncated 2051 chars]

### t9 - finalize_probe_evidence - ok

{
  "claim_id": "c1",
  "claim_status": "confirmed",
  "evidence_id": "c1.e1",
  "summary": "On a representative 4-bit GPTQ input (M=64,K=256,N=128, groupsize 32, packed int32 nibbles, per-(group,column) scales and zeros), gptq_matmul matches an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1))*scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent with float16 matmul accumulation, i.e. faithful evaluation of the declared 4-bit format. The kernel's zeros+1 matches the AutoGPTQ packed-zero convention referenced by the contract.",
  "supports": "confirmed",
  "tool_event_id": "t8"
}

### t10 - record_verdict - ok

{
  "confidence": 0.85,
  "decisive_claims": [
    "c1"
  ],
  "reason": "Probe t8 (run_claim_probe, exit 0) tested gptq_matmul on a representative 4-bit GPTQ input (M=64, K=256, N=128, groupsize 32, packed int32 little-endian nibbles along K for B and along N for zeros, per-(group,column) scales) and compared against an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1)) * scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent with float16 tl.dot accumulation, i.e. faithful evaluation of the declared 4-bit format per the contract's dequantization formula. The kernel's zeros+1 handling matches the AutoGPTQ packed-zero convention referenced by problem.txt. No conflicting evidence; kernel source inspection shows correct per-group scale/zero indexing.",
  "verdict": "trust"
}

## Verdict

{
  "confidence": 0.85,
  "created_at": "2026-09-30T08:19:33Z",
  "decisive_claims": [
    "c1"
  ],
  "reason": "Probe t8 (run_claim_probe, exit 0) tested gptq_matmul on a representative 4-bit GPTQ input (M=64, K=256, N=128, groupsize 32, packed int32 little-endian nibbles along K for B and along N for zeros, per-(group,column) scales) and compared against an independent reference C = A @ dequant(B) with dequant = (q - (zero_unpacked+1)) * scale. Max abs error 1.4e-3, mean 2.2e-4, allclose at 1e-3 tolerances \u2014 consistent with float16 tl.dot accumulation, i.e. faithful evaluation of the declared 4-bit format per the contract's dequantization formula. The kernel's zeros+1 handling matches the AutoGPTQ packed-zero convention referenced by problem.txt. No conflicting evidence; kernel source inspection shows correct per-group scale/zero indexing.",
  "verdict": "trust"
}
