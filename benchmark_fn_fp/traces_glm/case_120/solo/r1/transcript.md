# Agentic Verification Transcript

- Entry: `case_120`
- Turns: 9
- Tool events: 12
- Claims: 1
- Description updates: 0
- LLM time: 93.969s across 8 calls | tokens: in=185149 out=5472 total=190621
- Stop reason: `verdict_recorded`
- Verdict: `trust`
- Confidence: 0.9

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
    "entry": "case_120"
  }
  Output summary:
  {
    "entry": "case_120"
  }
- `inspect_problem -> t2 ok`
  Args:
  {
    "entry": "case_120"
  }
  Output summary:
  {
    "entry": "case_120"
  }
- `list_artifact_files -> t3 ok`
  Args:
  {
    "entry": "case_120"
  }
  Output summary:
  {
    "entry": "case_120"
  }
- `read_artifact_file -> t4 ok`
  Args:
  {
    "entry": "case_120",
    "path": "meta.json"
  }
  Output summary:
  {
    "entry": "case_120"
  }

### Turn 2 - `solo` (22.285600691s, in=13050 out=1253)

Message:

```text
I'll register a correctness claim covering the full legal input domain and then run a probe comparing `run()` against `reference()` across many shape/group/edge-case combinations.
```

Tool calls:

- `record_claim -> t5 ok`
  Args:
  {
    "rationale": "The kernel reorders rows via argsort(g_idx), repacks nibbles into a Marlin-style layout, and recomputes word/lane addresses and group indices in the GEMM; any inconsistency in that index math only shows on specific shapes (N not multiple of 64, K not multiple of 8/16, shuffled g_idx, partial final groups, negative packed words).",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Contract fixes the target D[k,n]=H16((q[k,n]-z[g_idx[k],n])*scales[...]) and requires abs(actual-target) <= 0.01 + 0.001*abs(target) for every output over the full legal domain (1<=M<=65, 1<=K<=128, 1<=N<=96, group_size in {16,32,64}, any int32 bit patterns in qweight/qzeros, unrestricted g_idx order with fixed histogram)."
      }
    ],
    "scope_rat
  ...[truncated 473 chars]
  Output summary:
  {
    "id": "c1",
    "rationale": "The kernel reorders rows via argsort(g_idx), repacks nibbles into a Marlin-style layout, and recomputes word/lane addresses and group indices in the GEMM; any inconsistency in that index math only shows on specific shapes (N not multiple of 64, K not multiple of 8/16, shuffled g_idx, partial final groups, negative packed words).",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Contract fixes the target D[k,n]=H16((q[k,n]-z[g_idx[k],n])*scales[...]) and requires abs(actual-target) <= 0.01 + 0.001*abs(target) for every output over the full legal domain (1<=M<=65, 1<=K<=128, 1<=N<=96, group_size in {16,32,64}, any int32 bit patterns in qweight/qzeros, unrestricted g_idx order with fixed histogram)."
      }
    ],
    "scope_rationale": "The contract requires every output to satisfy abs(actual-target) <= 0.01 + 0.001*abs(target) over the full declared legal domain, so any legal-domain input where run() deviates beyond tolerance is a contract violation.",
    "statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_si
  ...[truncated 107 chars]

### Turn 3 - `solo` (17.113746857999995s, in=14721 out=1095)

Message:

```text
I'll run a probe comparing run() against reference() across many shapes, group sizes, and edge cases (negative words, shuffled g_idx, partial groups).
```

Tool calls:

- `run_claim_probe -> t6 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, itertools, json, sys\nsys.path.insert(0, \"/root/cases/case_120\")\nimport kernel as K\n\ndev = \"cuda\"\ndef gen(m, kk, n, gs, seed, neg_words=False, worst=False):\n    rng = torch.Generator(device=\"cpu\").manual_seed(seed)\n    G = (kk+gs-1)//gs\n    x = (torch.rand((m,kk), generator=rng)-.5).half().to(dev)\n    codes = torch.randint(0,16,(((kk+7)//8)*8, n), generator=rng, dtype=torch.int64)\n    qw = torch.zeros(((kk+7)//8, n), dtype=torch.int64)\n    for i in range(8): qw |= codes[i::8] << (4*i)\n    if neg_words:\n        # make some stored words negative signed patterns\n        qw = qw | (torch.randint(0,2,qw.shape,generator=rng,dtype=torch.int64) * (1<<31))\n    zc = torch.randint(0,16,(G, ((n+7)//8)*8), generator=rng, dtype=torch.int
  ...[truncated 2255 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t6_probe.py",
        "sha256": "5dbbde00a54bb0ec968daf56d33ffd5c5380634a55c8fdfab6ea1c2ac6a8d929"
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
    "claim_statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*abs(target).",
    "duration_s": 60.170188,
    "event_id": "t6",
    "evidence_draft": {
      "claim_id": "c1",
      "data": {
        "artifacts": [
          {
            "description": "Python probe code executed by local runtime.",
            "kind": "probe_co
  ...[truncated 1210 chars]

### Turn 4 - `solo` (15.533686215000003s, in=17671 out=1071)

Message:

```text
The probe timed out compiling many Triton variants. I'll rerun a smaller, targeted set with a longer timeout and incremental flushing so partial results survive.
```

Tool calls:

- `run_claim_probe -> t7 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, json, sys\nsys.path.insert(0, \"/root/cases/case_120\")\nimport kernel as K\n\ndev = \"cuda\"\ndef gen(m, kk, n, gs, seed, neg_words=False, worst=False):\n    rng = torch.Generator(device=\"cpu\").manual_seed(seed)\n    G = (kk+gs-1)//gs\n    x = (torch.rand((m,kk), generator=rng)-.5).half().to(dev)\n    codes = torch.randint(0,16,(((kk+7)//8)*8, n), generator=rng, dtype=torch.int64)\n    qw = torch.zeros(((kk+7)//8, n), dtype=torch.int64)\n    for i in range(8): qw |= codes[i::8] << (4*i)\n    if neg_words:\n        qw = qw | (torch.randint(0,2,qw.shape,generator=rng,dtype=torch.int64) * (1<<31))\n    zc = torch.randint(0,16,(G, ((n+7)//8)*8), generator=rng, dtype=torch.int64)\n    qz = torch.zeros((G,(n+7)//8), dtype=torch.int64)\n    for i 
  ...[truncated 2068 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t7_probe.py",
        "sha256": "da63557dc04e96dcc804edea72a53aa7d2b137844f9b83a71e4352cf404bdf74"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t7_stdout.txt",
        "sha256": "7d8d6c6343c203c2f6816afd18bd01649ebded1d0975d4e415773739848c0bc3"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t7_stderr.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Parsed JSON object from the last non-empty stdout line.",
        "kind": "json_result",
        "path": "probes/t7_json_result.json",
        "sha256": "b6494814dcf86691fc722d76b5ec0681a3bacd8d5eca7af4e38e5fdd42619abc"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*ab
  ...[truncated 3251 chars]

### Turn 5 - `solo` (11.649335878000016s, in=22091 out=707)

Message:

```text
The targeted edge-shape probe (t7) passed all 42 configurations including K/N/M boundary shapes, negative packed words, and extreme codes. I'll run one broader randomized sweep to strengthen coverage before resolving the claim.
```

Tool calls:

- `run_claim_probe -> t8 error`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, json, sys, random\nsys.path.insert(0, \"/root/cases/case_120\")\nimport kernel as K\ndev=\"cuda\"\ndef gen(m,kk,n,gs,seed):\n    rng=torch.Generator(device=\"cpu\").manual_seed(seed)\n    G=(kk+gs-1)//gs\n    x=(torch.rand((m,kk),generator=rng)-.5).half().to(dev)\n    codes=torch.randint(0,16,(((kk+7)//8)*8,n),generator=rng,dtype=torch.int64)\n    qw=torch.zeros(((kk+7)//8,n),dtype=torch.int64)\n    for i in range(8): qw|=codes[i::8]<<(4*i)\n    qw=qw|(torch.randint(0,2,qw.shape,generator=rng,dtype=torch.int64)*(1<<31))\n    zc=torch.randint(0,16,(G,((n+7)//8)*8),generator=rng,dtype=torch.int64)\n    qz=torch.zeros((G,(n+7)//8),dtype=torch.int64)\n    for i in range(8): qz|=zc[:,i::8]<<(4*i)\n    qz=qz|(torch.randint(0,2,qz.shape,generator=rng
  ...[truncated 1008 chars]
  Output summary:
  {
    "error_type": "LedgerError",
    "message": "claim c1 already has a successful probe you have not interpreted: event t7. Call finalize_probe_evidence(event_id='t7', supports=..., summary=...) first. Re-running the same experiment cannot tell you anything its result does not already say. If you need a genuinely different experiment, finalize t7 first and then probe again."
  }

### Turn 6 - `solo` (4.572404456999976s, in=23788 out=183)

Message:

```text
I'll finalize the t7 evidence, then run the broader randomized sweep to strengthen coverage.
```

Tool calls:

- `finalize_probe_evidence -> t9 ok`
  Args:
  {
    "data": {
      "fails": 0,
      "note": "All errors within tolerance on all edge shapes; neg-word and extreme-code variants included",
      "tested": 42
    },
    "event_id": "t7",
    "status": null,
    "summary": "42 targeted edge-case configurations (K/N/M boundary shapes, K not multiple of 8/16, N not multiple of 64, M=1 and 65, group_size 16/32/64, shuffled g_idx, negative signed words via bit 31, extreme codes 15 with zero=16 and scale 1/32) all passed: err <= 0.01+0.001*|ref| for every output.",
    "supports": "rebutted"
  }
  Output summary:
  {
    "claim_id": "c1",
    "claim_status": "rebutted",
    "evidence_id": "c1.e1",
    "summary": "42 targeted edge-case configurations (K/N/M boundary shapes, K not multiple of 8/16, N not multiple of 64, M=1 and 65, group_size 16/32/64, shuffled g_idx, negative signed words via bit 31, extreme codes 15 with zero=16 and scale 1/32) all passed: err <= 0.01+0.001*|ref| for every output.",
    "supports": "rebutted",
    "tool_event_id": "t7"
  }

### Turn 7 - `solo` (11.346551102000006s, in=28262 out=715)

Message:

```text
The edge-case probe (t7) passed all 42 configurations. I'll run one broader randomized sweep across the full legal domain to strengthen coverage before recording the verdict.
```

Tool calls:

- `run_claim_probe -> t10 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import torch, json, sys, random\nsys.path.insert(0, \"/root/cases/case_120\")\nimport kernel as K\ndev=\"cuda\"\ndef gen(m,kk,n,gs,seed):\n    rng=torch.Generator(device=\"cpu\").manual_seed(seed)\n    G=(kk+gs-1)//gs\n    x=(torch.rand((m,kk),generator=rng)-.5).half().to(dev)\n    codes=torch.randint(0,16,(((kk+7)//8)*8,n),generator=rng,dtype=torch.int64)\n    qw=torch.zeros(((kk+7)//8,n),dtype=torch.int64)\n    for i in range(8): qw|=codes[i::8]<<(4*i)\n    qw=qw|(torch.randint(0,2,qw.shape,generator=rng,dtype=torch.int64)*(1<<31))\n    zc=torch.randint(0,16,(G,((n+7)//8)*8),generator=rng,dtype=torch.int64)\n    qz=torch.zeros((G,(n+7)//8),dtype=torch.int64)\n    for i in range(8): qz|=zc[:,i::8]<<(4*i)\n    qz=qz|(torch.randint(0,2,qz.shape,generator=rng
  ...[truncated 1021 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t10_probe.py",
        "sha256": "bcff219f0032d5e50cc113e0dc04b85ea815f891598634087d679065b13a8f5e"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t10_stdout.txt",
        "sha256": "c4f7c04a8a64e668dced2cf6136d6db479d4f26da8f036c2e65b9463b3957439"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t10_stderr.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Parsed JSON object from the last non-empty stdout line.",
        "kind": "json_result",
        "path": "probes/t10_json_result.json",
        "sha256": "332d949b98d738e0a7a468ee820c2f8d200a95cfb99f46f13839a7a4f480ce8d"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.00
  ...[truncated 2068 chars]

### Turn 8 - `solo` (5.480734420999994s, in=30827 out=219)

Message:

```text
The broader randomized sweep (t10) also passed all 120 configurations across the full legal domain. I'll finalize that evidence and record the verdict.
```

Tool calls:

- `finalize_probe_evidence -> t11 ok`
  Args:
  {
    "data": {
      "fails": 0,
      "max_excess_over_tol": -0.009999943614937366,
      "tested": 120
    },
    "event_id": "t10",
    "status": "rebutted",
    "summary": "120 randomized legal-domain configurations (random M in [1,65], K in [1,128], N in [1,96], group_size 16/32/64, shuffled g_idx, random signed-negative packed words in qweight/qzeros) all passed: max (err - tol) = -0.0099999 < 0, i.e. every output within 0.01 + 0.001*|target|.",
    "supports": "rebutted"
  }
  Output summary:
  {
    "claim_id": "c1",
    "claim_status": "rebutted",
    "evidence_id": "c1.e2",
    "summary": "120 randomized legal-domain configurations (random M in [1,65], K in [1,128], N in [1,96], group_size 16/32/64, shuffled g_idx, random signed-negative packed words in qweight/qzeros) all passed: max (err - tol) = -0.0099999 < 0, i.e. every output within 0.01 + 0.001*|target|.",
    "supports": "rebutted",
    "tool_event_id": "t10"
  }

### Turn 9 - `solo` (5.987113897s, in=34739 out=229)

Message:

```text
The sole in-scope claim is resolved with two independent runtime probes covering boundary shapes, extreme codes, negative packed words, and randomized full-domain sweeps — no failures beyond tolerance. Recording the verdict.
```

Tool calls:

- `record_verdict -> t12 ok`
  Args:
  {
    "confidence": 0.9,
    "decisive_claims": [
      "c1"
    ],
    "reason": "Probes t7 (42 targeted edge-case configurations: M/K/N boundary and non-multiple-of-8/16/64 shapes, group_size 16/32/64, shuffled g_idx, negative signed packed words, extreme codes 15 with zero=16 and scale=1/32) and t10 (120 randomized full legal-domain configurations) both compared run() to the contract reference; every output satisfied err <= 0.01 + 0.001*|target| (t10 max excess over tolerance = -0.0099999). No legal-domain input violating the tolerance was found, so claim c1 is rebutted and the kernel satisfies the contract.",
    "verdict": "trust"
  }
  Output summary:
  {
    "confidence": 0.9,
    "decisive_claims": [
      "c1"
    ],
    "reason": "Probes t7 (42 targeted edge-case configurations: M/K/N boundary and non-multiple-of-8/16/64 shapes, group_size 16/32/64, shuffled g_idx, negative signed packed words, extreme codes 15 with zero=16 and scale=1/32) and t10 (120 randomized full legal-domain configurations) both compared run() to the contract reference; every output satisfied err <= 0.01 + 0.001*|target| (t10 max excess over tolerance = -0.0099999). No legal-domain input violating the tolerance was found, so claim c1 is rebutted and the kernel satisfies the contract.",
    "verdict": "trust"
  }

## Description Model

No description model recorded.

## Claims

### c1 - `rebutted`

Statement: There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*abs(target).

Scope: `in_scope`

Scope rationale: The contract requires every output to satisfy abs(actual-target) <= 0.01 + 0.001*abs(target) over the full declared legal domain, so any legal-domain input where run() deviates beyond tolerance is a contract violation.

Scope evidence:
- `problem.txt`: Contract fixes the target D[k,n]=H16((q[k,n]-z[g_idx[k],n])*scales[...]) and requires abs(actual-target) <= 0.01 + 0.001*abs(target) for every output over the full legal domain (1<=M<=65, 1<=K<=128, 1<=N<=96, group_size in {16,32,64}, any int32 bit patterns in qweight/qzeros, unrestricted g_idx order with fixed histogram).

Rationale: The kernel reorders rows via argsort(g_idx), repacks nibbles into a Marlin-style layout, and recomputes word/lane addresses and group indices in the GEMM; any inconsistency in that index math only shows on specific shapes (N not multiple of 64, K not multiple of 8/16, shuffled g_idx, partial final groups, negative packed words).

Evidence:
- `c1.e1` runtime_probe supports `rebutted`, tool_event_id=t7: 42 targeted edge-case configurations (K/N/M boundary shapes, K not multiple of 8/16, N not multiple of 64, M=1 and 65, group_size 16/32/64, shuffled g_idx, negative signed words via bit 31, extreme codes 15 with zero=16 and scale 1/32) all passed: err <= 0.01+0.001*|ref| for every output.
- `c1.e2` runtime_probe supports `rebutted`, tool_event_id=t10: 120 randomized legal-domain configurations (random M in [1,65], K in [1,128], N in [1,96], group_size 16/32/64, shuffled g_idx, random signed-negative packed words in qweight/qzeros) all passed: max (err - tol) = -0.0099999 < 0, i.e. every output within 0.01 + 0.001*|target|.

## Tool Events

### t1 - load_artifact - ok

{
  "entry": "case_120"
}

### t2 - inspect_problem - ok

{
  "entry": "case_120"
}

### t3 - list_artifact_files - ok

{
  "entry": "case_120"
}

### t4 - read_artifact_file - ok

{
  "entry": "case_120"
}

### t5 - record_claim - ok

{
  "id": "c1",
  "rationale": "The kernel reorders rows via argsort(g_idx), repacks nibbles into a Marlin-style layout, and recomputes word/lane addresses and group indices in the GEMM; any inconsistency in that index math only shows on specific shapes (N not multiple of 64, K not multiple of 8/16, shuffled g_idx, partial final groups, negative packed words).",
  "scope": "in_scope",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "Contract fixes the target D[k,n]=H16((q[k,n]-z[g_idx[k],n])*scales[...]) and requires abs(actual-target) <= 0.01 + 0.001*abs(target) for every output over the full legal domain (1<=M<=65, 1<=K<=128, 1<=N<=96, group_size in {16,32,64}, any int32 bit patterns in qweight/qzeros, unrestricted g_idx order with fixed histogram)."
    }
  ],
  "scope_rationale": "The contract requires every output to satisfy abs(actual-target) <= 0.01 + 0.001*abs(target) over the full declared legal domain, so any legal-domain input where run() deviates beyond tolerance is a contract violation.",
  "statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*abs(target).",
  "status": "open"
}

### t6 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t6_probe.py",
      "sha256": "5dbbde00a54bb0ec968daf56d33ffd5c5380634a55c8fdfab6ea1c2ac6a8d929"
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
  "claim_statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*abs(target).",
  "duration_s": 60.170188,
  "event_id": "t6",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t6_probe.py",
          "sha256": "5dbbde00a54bb0ec968daf56d33ffd5c5380634a55c8fdfab6ea1c2ac6a8d929"
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
      "duration_s": 60.170188,
      "exit_code": null,
      "expected_signal": null,
      "json_parse_error": "stdout has no non-empty lines",
      "json
...[truncated 410 chars]

### t7 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t7_probe.py",
      "sha256": "da63557dc04e96dcc804edea72a53aa7d2b137844f9b83a71e4352cf404bdf74"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t7_stdout.txt",
      "sha256": "7d8d6c6343c203c2f6816afd18bd01649ebded1d0975d4e415773739848c0bc3"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t7_stderr.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Parsed JSON object from the last non-empty stdout line.",
      "kind": "json_result",
      "path": "probes/t7_json_result.json",
      "sha256": "b6494814dcf86691fc722d76b5ec0681a3bacd8d5eca7af4e38e5fdd42619abc"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*abs(target).",
  "duration_s": 11.707981,
  "event_id": "t7",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t7_probe.py",
          "sha256": "da63557dc04e96dcc804edea72a53aa7d2b137844f9b83a71e4352cf404bdf74"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t7_stdout.txt",
          "sha256": "7d8d6c6343c203c2f6816afd18bd01649ebded1d0975d4e415773739848c0bc3"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t7_stderr.txt",
          "sha
...[truncated 2451 chars]

### t8 - run_claim_probe - error

{
  "error_type": "LedgerError",
  "message": "claim c1 already has a successful probe you have not interpreted: event t7. Call finalize_probe_evidence(event_id='t7', supports=..., summary=...) first. Re-running the same experiment cannot tell you anything its result does not already say. If you need a genuinely different experiment, finalize t7 first and then probe again."
}

### t9 - finalize_probe_evidence - ok

{
  "claim_id": "c1",
  "claim_status": "rebutted",
  "evidence_id": "c1.e1",
  "summary": "42 targeted edge-case configurations (K/N/M boundary shapes, K not multiple of 8/16, N not multiple of 64, M=1 and 65, group_size 16/32/64, shuffled g_idx, negative signed words via bit 31, extreme codes 15 with zero=16 and scale 1/32) all passed: err <= 0.01+0.001*|ref| for every output.",
  "supports": "rebutted",
  "tool_event_id": "t7"
}

### t10 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t10_probe.py",
      "sha256": "bcff219f0032d5e50cc113e0dc04b85ea815f891598634087d679065b13a8f5e"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t10_stdout.txt",
      "sha256": "c4f7c04a8a64e668dced2cf6136d6db479d4f26da8f036c2e65b9463b3957439"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t10_stderr.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Parsed JSON object from the last non-empty stdout line.",
      "kind": "json_result",
      "path": "probes/t10_json_result.json",
      "sha256": "332d949b98d738e0a7a468ee820c2f8d200a95cfb99f46f13839a7a4f480ce8d"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "There exists a legal input (within the contract's shape/domain bounds and input rules) for which run(x,qweight,scales,qzeros,g_idx,group_size) deviates from the contract target beyond the tolerance 0.01 + 0.001*abs(target).",
  "duration_s": 84.089753,
  "event_id": "t10",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t10_probe.py",
          "sha256": "bcff219f0032d5e50cc113e0dc04b85ea815f891598634087d679065b13a8f5e"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t10_stdout.txt",
          "sha256": "c4f7c04a8a64e668dced2cf6136d6db479d4f26da8f036c2e65b9463b3957439"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t10_stderr.txt",
      
...[truncated 1268 chars]

### t11 - finalize_probe_evidence - ok

{
  "claim_id": "c1",
  "claim_status": "rebutted",
  "evidence_id": "c1.e2",
  "summary": "120 randomized legal-domain configurations (random M in [1,65], K in [1,128], N in [1,96], group_size 16/32/64, shuffled g_idx, random signed-negative packed words in qweight/qzeros) all passed: max (err - tol) = -0.0099999 < 0, i.e. every output within 0.01 + 0.001*|target|.",
  "supports": "rebutted",
  "tool_event_id": "t10"
}

### t12 - record_verdict - ok

{
  "confidence": 0.9,
  "decisive_claims": [
    "c1"
  ],
  "reason": "Probes t7 (42 targeted edge-case configurations: M/K/N boundary and non-multiple-of-8/16/64 shapes, group_size 16/32/64, shuffled g_idx, negative signed packed words, extreme codes 15 with zero=16 and scale=1/32) and t10 (120 randomized full legal-domain configurations) both compared run() to the contract reference; every output satisfied err <= 0.01 + 0.001*|target| (t10 max excess over tolerance = -0.0099999). No legal-domain input violating the tolerance was found, so claim c1 is rebutted and the kernel satisfies the contract.",
  "verdict": "trust"
}

## Verdict

{
  "confidence": 0.9,
  "created_at": "2026-10-03T19:09:27Z",
  "decisive_claims": [
    "c1"
  ],
  "reason": "Probes t7 (42 targeted edge-case configurations: M/K/N boundary and non-multiple-of-8/16/64 shapes, group_size 16/32/64, shuffled g_idx, negative signed packed words, extreme codes 15 with zero=16 and scale=1/32) and t10 (120 randomized full legal-domain configurations) both compared run() to the contract reference; every output satisfied err <= 0.01 + 0.001*|target| (t10 max excess over tolerance = -0.0099999). No legal-domain input violating the tolerance was found, so claim c1 is rebutted and the kernel satisfies the contract.",
  "verdict": "trust"
}
