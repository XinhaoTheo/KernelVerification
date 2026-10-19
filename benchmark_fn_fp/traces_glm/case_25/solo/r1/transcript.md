# Agentic Verification Transcript

- Entry: `case_26`
- Turns: 7
- Tool events: 11
- Claims: 1
- Description updates: 0
- LLM time: 75.015s across 6 calls | tokens: in=82562 out=6780 total=89342
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
    "entry": "case_26"
  }
  Output summary:
  {
    "entry": "case_26"
  }
- `inspect_problem -> t2 ok`
  Args:
  {
    "entry": "case_26"
  }
  Output summary:
  {
    "entry": "case_26"
  }
- `list_artifact_files -> t3 ok`
  Args:
  {
    "entry": "case_26"
  }
  Output summary:
  {
    "entry": "case_26"
  }
- `read_artifact_file -> t4 ok`
  Args:
  {
    "entry": "case_26",
    "path": "meta.json"
  }
  Output summary:
  {
    "entry": "case_26"
  }

### Turn 2 - `solo` (37.718914201s, in=6693 out=3178)

Message:

```text
Binary search on threshold: after loop, lo is the threshold such that count(x>=lo) >= K... Actually lo converges to a value where count >= K and count(x>=hi) < K. Keep = x >= lo. Ties at lo admitted — fine per contract. Concern: with 40 iterations and float precision, lo may be slightly below true kth value, admitting extra entries larger than dropped ones? Requirement: every kept entry ≥ every dropped one. If lo slightly below the kth value, entries between lo and kth value get kept but are ≥ dropped ones — still fine? Dropped entries are those < lo, kept are ≥ lo, so trivially kept ≥ dropped. Wait — the invariant is keep = x >= lo, drop = x < lo, so kept ≥ dropped by construction. The real invariant is guaranteed by binary search? Binary search: count(x>=mid) >= K → lo=mid, else hi=mid. After convergence lo ≈ threshold where count(x>=lo) ≥ K. Kept = {x ≥ lo}, which includes all x > lo... any x < lo is dropped, and all kept x ≥ lo ≥ dropped x. So invariant holds trivially regardless of search accuracy — unless lo fails to converge below hi (floating point stall: mid == lo or hi). If mid = (lo+hi)/2 rounds to lo (when hi≈lo), then if count>=K, lo=mid=lo, hi=mid=lo... fine. If count
...[truncated 9870 chars]
```

Tool calls:

- `record_claim -> t5 ok`
  Args:
  {
    "rationale": "Bisection on floats can stall (mid rounds to lo or hi) leaving lo strictly below the true k-th value, so entries in [lo, v_k) with distinct values get kept although they are not among the top-k and not tied with the k-th entry, which the contract's tie relaxation does not cover.",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Contract requires out[j]=scores[j] only for entries among the k largest, with ties at the k-th position as the only admitted extra-keeping case."
      }
    ],
    "scope_rationale": "problem.txt states kept entries must be among the k largest, with only ties at the k-th position admitted; a kept entry strictly below the k-th value with distinct dropped values would violate the only stated acceptance condition's spirit and t
  ...[truncated 321 chars]
  Output summary:
  {
    "id": "c1",
    "rationale": "Bisection on floats can stall (mid rounds to lo or hi) leaving lo strictly below the true k-th value, so entries in [lo, v_k) with distinct values get kept although they are not among the top-k and not tied with the k-th entry, which the contract's tie relaxation does not cover.",
    "scope": "unknown",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Contract requires out[j]=scores[j] only for entries among the k largest, with ties at the k-th position as the only admitted extra-keeping case."
      }
    ],
    "scope_rationale": "problem.txt states kept entries must be among the k largest, with only ties at the k-th position admitted; a kept entry strictly below the k-th value with distinct dropped values would violate the only stated acceptance condition's spirit and the top-k definition on inputs the benchmark admits (ordinary score rows).",
    "statement": "The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with distinct values.",
    "status": "open"
  }

### Turn 3 - `solo` (18.441418185000003s, in=8440 out=1742)

Message:

```text
Key concern: keep = x >= lo. If lo < kth value (bisection stalls), entries strictly between lo and kth value get kept... wait, is that possible? lo converges to threshold t where count(x>=t)>=K. Kept set = {x >= lo}. If lo < v_k (true kth value), then entries with value in [lo, v_k) are kept while some dropped entries (x < lo) are smaller — so kept ≥ dropped holds. The contract's requirement is only "every kept entry ≥ every dropped one". Kept set = {x>=lo}, dropped = {x<lo}, so invariant holds by construction regardless of convergence! The only failure mode: if lo > true kth value, then a dropped entry could be ≥ kept... no, dropped = x<lo, kept = x>=lo, so kept ≥ dropped always. So the invariant trivially holds. Extra kept entries beyond top-k? If lo is slightly below v_k, extra kept entries are those in [lo, v_k) — but wait, could lo be below the (k+1)th value while kth value is above lo? Then dropped entries (x<lo) are smaller than kept entries. Contract's stated acceptance: "every kept entry is at least as large as every dropped one" — satisfied. But also out[j]=scores[j] only if among k largest — kept entries between lo and v_k are not among top-k. Hmm, but the contract's tie
...[truncated 3703 chars]
```

Tool calls:

- `run_claim_probe -> t6 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import json, torch, sys\nsys.path.insert(0, \"/root/cases/case_26\")\nfrom kernel import topk_mask\n\ntorch.manual_seed(0)\nresults = []\nworst = None\nfor trial, (N, K, dist) in enumerate([(256, 10, \"randn\"), (1024, 50, \"randn\"),\n                                      (64, 32, \"clustered\"), (256, 10, \"tied\"),\n                                      (256, 10, \"cutoff_cluster\")]):\n    if dist == \"randn\":\n        s = torch.randn(64, N, device=\"cuda\")\n    elif dist == \"clustered\":\n        s = torch.randn(64, N, device=\"cuda\") * 0.01\n    elif dist == \"tied\":\n        s = torch.randint(0, 5, (64, N), device=\"cuda\").float()\n    else:\n        s = torch.randn(64, N, device=\"cuda\")\n        s[:, 5:40] = s[:, 5]  # many values equal/adja
  ...[truncated 1166 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t6_probe.py",
        "sha256": "f03b5b599db1d6636a73a58052bc933ecdb8dae91b276d285a5fb400bc36fd76"
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
        "sha256": "9d250b49d19f62bf76275aed802c1462c89f0c7af170cf2f0910822ee997586f"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with distinct values.",
    "duration_s": 8.283364,
    "event_id": "t6",
    "evidence_draft": {
      "claim_id": "c1",
      "data": {
        "artifacts": [
          {
            "description": "Python probe code executed by local runtime.",
            "kind": "probe_
  ...[truncated 1994 chars]

### Turn 4 - `solo` (7.149751088000002s, in=11307 out=862)

Message:

```text
Fix probe bug: s[:,5:40] assignment broadcast issue — need s[:,5:40] = s[:,5:6]. Also add a stress case with many distinct values very close together to test bisection precision: e.g. values generated by kth*(1+eps). Rerun probe on c1.The previous probe crashed on a broadcasting bug (`s[:, 5]` didn't expand). I'll fix that and also add a targeted case with many distinct but extremely close values near the cutoff, which is exactly the input regime where float bisection could stall with `lo` strictly below the true k-th value.
```

Tool calls:

- `run_claim_probe -> t7 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import json, torch, sys\nsys.path.insert(0, \"/root/cases/case_26\")\nfrom kernel import topk_mask\n\ntorch.manual_seed(0)\nresults = []\nworst = []\nfor N, K, dist in [(256, 10, \"randn\"), (1024, 50, \"randn\"),\n                   (64, 32, \"clustered\"), (256, 10, \"tied\"),\n                   (256, 10, \"cutoff_cluster\"),\n                   (256, 10, \"eps_close\"),\n                   (256, 128, \"k_half\")]:\n    if dist == \"randn\":\n        s = torch.randn(64, N, device=\"cuda\")\n    elif dist == \"clustered\":\n        s = torch.randn(64, N, device=\"cuda\") * 0.01\n    elif dist == \"tied\":\n        s = torch.randint(0, 5, (64, N), device=\"cuda\").float()\n    elif dist == \"cutoff_cluster\":\n        s = torch.randn(64, N, device=\"cuda\"
  ...[truncated 1792 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t7_probe.py",
        "sha256": "d04ce1e463eb9128cd2dacc6da66caad7901e1775aeb501afd90e63b03be51d9"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t7_stdout.txt",
        "sha256": "082e8442816e82ed4001fad6788a18060725efcbd90a98bdb7e4c295274f15b1"
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
        "sha256": "6df8bc88870c26a19cd9fad8affe2cdcfa74fc015063354e8fddca44fa32cc29"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with dis
  ...[truncated 7874 chars]

### Turn 5 - `solo` (5.719308064000003s, in=16119 out=604)

Message:

```text
Probe t7 passed across all cases. Also reasoning: keep = x >= lo, dropped = x < lo, so dominance holds by construction; only issue is keeping untied entries below kth — not observed, and even conceptually lo ≥ ... Actually if lo < kth value, entries in [lo, v_k) kept untied — but those are ≥ all dropped. Contract says requirement is "every kept entry ≥ every dropped one" — that's the acceptance condition, so even untied extra keeps would satisfy the contract's stated requirement? The contract says both readings accepted "the requirement is that every kept entry is at least as large as every dropped one." So keep=x>=lo trivially satisfies. Contract satisfied. Rebut c1, finalize, verdict trust.The probe t7 ran clean across 7 distributions including ties, cutoff clusters, and adversarially close values: dominance invariant holds, all entries above the k-th value are kept, no untied extra keeps, failures = []. Note that keep = x >= lo and dropped = x < lo makes "kept ≥ dropped" hold by construction; the only residual risk (untied entries below the true k-th value being kept) was measured at 0 in every case, and even that reading satisfies the contract's stated acceptance condition. Rec
...[truncated 36 chars]
```

Tool calls:

- `finalize_probe_evidence -> t8 error`
  Args:
  {
    "event_id": "t7",
    "status": "confirmed",
    "summary": "Probe t7 (7 distributions incl. tied, cutoff-clustered, eps-close) shows: kept>=dropped dominance holds in all rows, all entries > kth are kept, and 0 untied entries below the kth value were kept (kept_count == K except in tie cases where extra keeps are exactly tied at the cutoff, which the contract admits). No case of lo stalling below the kth cutoff with distinct-value over-keeping was found.",
    "supports": "rebutted"
  }
  Output summary:
  {
    "error_type": "LedgerError",
    "message": "claim c1 needs evidence supporting confirmed before status update"
  }
- `record_verdict -> t9 error`
  Args:
  {
    "confidence": 0.9,
    "decisive_claims": [
      "c1"
    ],
    "reason": "The kernel's final mask is keep = x >= lo, dropped = x < lo, so the contract's acceptance condition (every kept entry >= every dropped entry) holds by construction regardless of bisection convergence. Runtime probe t7 over 7 input distributions (randn, small-scale clustered, integer-tied, many-equal-at-cutoff, extremely close distinct values, k=N/2) confirmed: dominance invariant holds in all rows, all entries strictly above the k-th value are kept, and the count of untied entries kept below the k-th value was 0 everywhere; kept counts equal K except in tie cases where extra keeps are exactly tied at the cutoff, an admitted reading. Claim c1 (bisection stall keeps untied entries below the kth value) is rebutted. Contr
  ...[truncated 39 chars]
  Output summary:
  {
    "error_type": "ValueError",
    "message": "trust verdict requires all claims to be resolved; open claims: c1"
  }

### Turn 6 - `solo` (1.838857800999989s, in=19026 out=98)

Message:

```text
The finalize_probe_evidence recorded evidence (c1.e1) with supports=rebutted, but the status update failed. I need to update_claim_status c1 → rebutted, then record verdict trust.The probe evidence for claim c1 is already recorded (event t7, supports=rebutted), but the claim status update and verdict failed. Closing both now.
```

Tool calls:

- `update_claim_status -> t10 ok`
  Args:
  {
    "claim_id": "c1",
    "status": "rebutted"
  }
  Output summary:
  {
    "id": "c1",
    "status": "rebutted"
  }

### Turn 7 - `solo` (4.146881176999997s, in=20977 out=296)

Message:

```text
Claim c1 is rebutted and resolved. Just record the verdict.Claim c1 is now resolved (rebutted) with probe evidence t7/c1.e1. The final mask keep = x ≥ lo makes the contract's dominance condition hold by construction, and the probe measured zero untied over-keeps across 7 distributions including tied and cutoff-clustered inputs. Recording the verdict.
```

Tool calls:

- `record_verdict -> t11 ok`
  Args:
  {
    "confidence": 0.9,
    "decisive_claims": [
      "c1"
    ],
    "reason": "The kernel's final mask is keep = x >= lo, dropped = x < lo, so the contract's acceptance condition (every kept entry >= every dropped entry) holds by construction regardless of bisection convergence. Runtime probe t7 (evidence c1.e1) over 7 input distributions (randn, small-scale clustered, integer-tied, many-equal-at-cutoff, extremely close distinct values, k=N/2) confirmed: dominance invariant holds in all rows, all entries strictly above the k-th value are kept, and the count of untied entries kept below the k-th value was 0 everywhere; kept counts equal K except in tie cases where extra keeps are exactly tied at the cutoff, which the contract explicitly admits. Claim c1 (bisection stall keeps untied entries below
  ...[truncated 73 chars]
  Output summary:
  {
    "confidence": 0.9,
    "decisive_claims": [
      "c1"
    ],
    "reason": "The kernel's final mask is keep = x >= lo, dropped = x < lo, so the contract's acceptance condition (every kept entry >= every dropped entry) holds by construction regardless of bisection convergence. Runtime probe t7 (evidence c1.e1) over 7 input distributions (randn, small-scale clustered, integer-tied, many-equal-at-cutoff, extremely close distinct values, k=N/2) confirmed: dominance invariant holds in all rows, all entries strictly above the k-th value are kept, and the count of untied entries kept below the k-th value was 0 everywhere; kept counts equal K except in tie cases where extra keeps are exactly tied at the cutoff, which the contract explicitly admits. Claim c1 (bisection stall keeps untied entries below the kth value) is rebutted. Contract satisfied.",
    "verdict": "trust"
  }

## Description Model

No description model recorded.

## Claims

### c1 - `rebutted`

Statement: The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with distinct values.

Scope: `unknown`

Scope rationale: problem.txt states kept entries must be among the k largest, with only ties at the k-th position admitted; a kept entry strictly below the k-th value with distinct dropped values would violate the only stated acceptance condition's spirit and the top-k definition on inputs the benchmark admits (ordinary score rows).

Scope evidence:
- `problem.txt`: Contract requires out[j]=scores[j] only for entries among the k largest, with ties at the k-th position as the only admitted extra-keeping case.

Rationale: Bisection on floats can stall (mid rounds to lo or hi) leaving lo strictly below the true k-th value, so entries in [lo, v_k) with distinct values get kept although they are not among the top-k and not tied with the k-th entry, which the contract's tie relaxation does not cover.

Evidence:
- `c1.e1` runtime_probe supports `rebutted`, tool_event_id=t7: Probe t7 (7 distributions incl. tied, cutoff-clustered, eps-close) shows: kept>=dropped dominance holds in all rows, all entries > kth are kept, and 0 untied entries below the kth value were kept (kept_count == K except in tie cases where extra keeps are exactly tied at the cutoff, which the contract admits). No case of lo stalling below the kth cutoff with distinct-value over-keeping was found.

## Tool Events

### t1 - load_artifact - ok

{
  "entry": "case_26"
}

### t2 - inspect_problem - ok

{
  "entry": "case_26"
}

### t3 - list_artifact_files - ok

{
  "entry": "case_26"
}

### t4 - read_artifact_file - ok

{
  "entry": "case_26"
}

### t5 - record_claim - ok

{
  "id": "c1",
  "rationale": "Bisection on floats can stall (mid rounds to lo or hi) leaving lo strictly below the true k-th value, so entries in [lo, v_k) with distinct values get kept although they are not among the top-k and not tied with the k-th entry, which the contract's tie relaxation does not cover.",
  "scope": "unknown",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "Contract requires out[j]=scores[j] only for entries among the k largest, with ties at the k-th position as the only admitted extra-keeping case."
    }
  ],
  "scope_rationale": "problem.txt states kept entries must be among the k largest, with only ties at the k-th position admitted; a kept entry strictly below the k-th value with distinct dropped values would violate the only stated acceptance condition's spirit and the top-k definition on inputs the benchmark admits (ordinary score rows).",
  "statement": "The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with distinct values.",
  "status": "open"
}

### t6 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t6_probe.py",
      "sha256": "f03b5b599db1d6636a73a58052bc933ecdb8dae91b276d285a5fb400bc36fd76"
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
      "sha256": "9d250b49d19f62bf76275aed802c1462c89f0c7af170cf2f0910822ee997586f"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with distinct values.",
  "duration_s": 8.283364,
  "event_id": "t6",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t6_probe.py",
          "sha256": "f03b5b599db1d6636a73a58052bc933ecdb8dae91b276d285a5fb400bc36fd76"
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
          "sha256": "9d250b49d19f62bf76275aed802c1462c89f0c7af170cf2f0910822ee997586f"
        }
      ],
      "duration_s": 8.283364,
      "exit_code": 1,
      "expected_signal": null,
      "json_parse_error": "stdout has no non-empty lines",
      "json_r
...[truncated 1194 chars]

### t7 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t7_probe.py",
      "sha256": "d04ce1e463eb9128cd2dacc6da66caad7901e1775aeb501afd90e63b03be51d9"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t7_stdout.txt",
      "sha256": "082e8442816e82ed4001fad6788a18060725efcbd90a98bdb7e4c295274f15b1"
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
      "sha256": "6df8bc88870c26a19cd9fad8affe2cdcfa74fc015063354e8fddca44fa32cc29"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "The kernel may keep entries strictly smaller than the k-th largest value (not tied with it), because the float bisection can leave lo strictly below the true k-th cutoff, yielding more than K kept entries with distinct values.",
  "duration_s": 5.684552,
  "event_id": "t7",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t7_probe.py",
          "sha256": "d04ce1e463eb9128cd2dacc6da66caad7901e1775aeb501afd90e63b03be51d9"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t7_stdout.txt",
          "sha256": "082e8442816e82ed4001fad6788a18060725efcbd90a98bdb7e4c295274f15b1"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t7_stderr.txt",
          "s
...[truncated 7074 chars]

### t8 - finalize_probe_evidence - error

{
  "error_type": "LedgerError",
  "message": "claim c1 needs evidence supporting confirmed before status update"
}

### t9 - record_verdict - error

{
  "error_type": "ValueError",
  "message": "trust verdict requires all claims to be resolved; open claims: c1"
}

### t10 - update_claim_status - ok

{
  "id": "c1",
  "status": "rebutted"
}

### t11 - record_verdict - ok

{
  "confidence": 0.9,
  "decisive_claims": [
    "c1"
  ],
  "reason": "The kernel's final mask is keep = x >= lo, dropped = x < lo, so the contract's acceptance condition (every kept entry >= every dropped entry) holds by construction regardless of bisection convergence. Runtime probe t7 (evidence c1.e1) over 7 input distributions (randn, small-scale clustered, integer-tied, many-equal-at-cutoff, extremely close distinct values, k=N/2) confirmed: dominance invariant holds in all rows, all entries strictly above the k-th value are kept, and the count of untied entries kept below the k-th value was 0 everywhere; kept counts equal K except in tie cases where extra keeps are exactly tied at the cutoff, which the contract explicitly admits. Claim c1 (bisection stall keeps untied entries below the kth value) is rebutted. Contract satisfied.",
  "verdict": "trust"
}

## Verdict

{
  "confidence": 0.9,
  "created_at": "2026-09-10T05:12:39Z",
  "decisive_claims": [
    "c1"
  ],
  "reason": "The kernel's final mask is keep = x >= lo, dropped = x < lo, so the contract's acceptance condition (every kept entry >= every dropped entry) holds by construction regardless of bisection convergence. Runtime probe t7 (evidence c1.e1) over 7 input distributions (randn, small-scale clustered, integer-tied, many-equal-at-cutoff, extremely close distinct values, k=N/2) confirmed: dominance invariant holds in all rows, all entries strictly above the k-th value are kept, and the count of untied entries kept below the k-th value was 0 everywhere; kept counts equal K except in tie cases where extra keeps are exactly tied at the cutoff, which the contract explicitly admits. Claim c1 (bisection stall keeps untied entries below the kth value) is rebutted. Contract satisfied.",
  "verdict": "trust"
}
