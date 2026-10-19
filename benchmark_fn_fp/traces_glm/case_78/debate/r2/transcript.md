# Agentic Verification Transcript

- Entry: `case_151`
- Turns: 7
- Tool events: 16
- Claims: 3
- Description updates: 1
- LLM time: 100.646s across 6 calls | tokens: in=171883 out=6600 total=178483
- Stop reason: `verdict_recorded`
- Verdict: `reject`
- Confidence: 0.97
- Skeptic review: `no_new_claims`

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
    "entry": "case_151"
  }
  Output summary:
  {
    "entry": "case_151"
  }
- `inspect_problem -> t2 ok`
  Args:
  {
    "entry": "case_151"
  }
  Output summary:
  {
    "entry": "case_151"
  }
- `list_artifact_files -> t3 ok`
  Args:
  {
    "entry": "case_151"
  }
  Output summary:
  {
    "entry": "case_151"
  }
- `read_artifact_file -> t4 ok`
  Args:
  {
    "entry": "case_151",
    "path": "meta.json"
  }
  Output summary:
  {
    "entry": "case_151"
  }

### Turn 2 - `describer` (26.200660807s, in=8200 out=1693)

Message:

```text
Analyze kernel and record description update.
```

Tool calls:

- `record_description_update -> t5 ok`
  Args:
  {
    "contract_model": [
      "Forward cyclic NTT per row: out[b,k] = sum_j x[b,j] * omega_N^(j*k) mod 3329, with omega_N = 17^(256/N) mod 3329, N in {64,128}, positive exponent, natural output order, no normalization.",
      "Exactness required: abs tol 0, rel tol 0, canonical residues in [0,3328].",
      "Domain: ordinary int32 CUDA tensor, shape [B,N] with 1<=B<=128, contiguous (nonzero storage offsets legal), all field values 0..3328 legal; no autograd/inference-mode flag issues; fresh output tensor preserving input.",
      "validate_inputs() enforces the full domain and is called by run(); reference() is a direct modular DFT (int64 products/sums) and accepts CPU.",
      "Single correctness verdict for whole implementation; any transform factorization, internal residue representation, or mod
  ...[truncated 6065 chars]
  Output summary:
  {
    "contract_model": [
      "Forward cyclic NTT per row: out[b,k] = sum_j x[b,j] * omega_N^(j*k) mod 3329, with omega_N = 17^(256/N) mod 3329, N in {64,128}, positive exponent, natural output order, no normalization.",
      "Exactness required: abs tol 0, rel tol 0, canonical residues in [0,3328].",
      "Domain: ordinary int32 CUDA tensor, shape [B,N] with 1<=B<=128, contiguous (nonzero storage offsets legal), all field values 0..3328 legal; no autograd/inference-mode flag issues; fresh output tensor preserving input.",
      "validate_inputs() enforces the full domain and is called by run(); reference() is a direct modular DFT (int64 products/sums) and accepts CPU.",
      "Single correctness verdict for whole implementation; any transform factorization, internal residue representation, or modular-reduction algorithm is allowed."
    ],
    "impact_on_claims": [],
    "kernel_model": [
      "Kernel does a bit-reversal load then radix-4 fused butterfly stages via tl.gather, using precomputed coefficient banks packed in _make_coefficients (lines 19-37), a Montgomery-style reduction in _multiply with R=2^16 (lines 41-45), and an optional final radix-2 stage when LOG_N is odd (lines 86-95).",
     
  ...[truncated 5755 chars]

### Turn 3 - `skeptic` (23.702613999s, in=20506 out=1822)

Message:

```text
The description model is complete and the ledger is empty. I'll record the three highest-risk, directly testable claims drawn from the risk map, all tied to the stated contract (exact forward cyclic NTT, tol 0, N∈{64,128}).
```

Tool calls:

- `record_claim -> t6 ok`
  Args:
  {
    "rationale": "The odd-LOG_N branch (lines 86-95) is structurally distinct from the fused radix-4 path and is exercised only at N=128: the coefficient start formula `5*((1<<(2*(LOG_N//2)))-1)//3`, the single conditional subtract in _multiply, and the (value+3329)%3329 final reduce all differ from the fused-stage path. Any mismatch in the final bank layout or the bit-reversal interplay at LOG_N=7 would corrupt every N=128 output while N=64 stays correct, and typical make_inputs examples use width=64.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "N is either 64 or 128 and every result must equal out[b,k] = sum x[b,j]*omega_N^(j*k) mod 3329 exactly with abs tol 0 and rel tol 0."
      }
    ],
    "scope_rationale": "The contract explicitly
  ...[truncated 616 chars]
  Output summary:
  {
    "id": "c1",
    "rationale": "The odd-LOG_N branch (lines 86-95) is structurally distinct from the fused radix-4 path and is exercised only at N=128: the coefficient start formula `5*((1<<(2*(LOG_N//2)))-1)//3`, the single conditional subtract in _multiply, and the (value+3329)%3329 final reduce all differ from the fused-stage path. Any mismatch in the final bank layout or the bit-reversal interplay at LOG_N=7 would corrupt every N=128 output while N=64 stays correct, and typical make_inputs examples use width=64.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "N is either 64 or 128 and every result must equal out[b,k] = sum x[b,j]*omega_N^(j*k) mod 3329 exactly with abs tol 0 and rel tol 0."
      }
    ],
    "scope_rationale": "The contract explicitly requires N=128 as a supported, exactly-correct length (\"N is either 64 or 128 ... Every result must equal the mathematical target exactly: absolute tolerance=0 and relative tolerance=0\"). The N=128 path (odd LOG_N=7, extra final radix-2 stage at lines 86-95 plus the appended coefficient bank at lines 35-36) must match the reference matrix pow(omega, j*k) exactly.",
    "statement": 
  ...[truncated 250 chars]
- `record_claim -> t7 ok`
  Args:
  {
    "rationale": "A twiddle exponent sign error or conjugated mapping is the classic structural bug in NTT kernels: the code compiles, bit-reversal and butterflies look plausible, but the result is the inverse/negacyclic transform. The quadrant-indexed sums/differences at lines 77-83 and the cross twiddles at lines 27-30 determine this and must be checked numerically against the direct DFT with a positive exponent.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Output position k is frequency k in natural order, with the positive exponent omega_N**(j*k); 'This is a cyclic transform, not a negacyclic transform'."
      }
    ],
    "scope_rationale": "The contract requires the forward transform with the POSITIVE exponent omega_N^(j*k) and natu
  ...[truncated 452 chars]
  Output summary:
  {
    "id": "c2",
    "rationale": "A twiddle exponent sign error or conjugated mapping is the classic structural bug in NTT kernels: the code compiles, bit-reversal and butterflies look plausible, but the result is the inverse/negacyclic transform. The quadrant-indexed sums/differences at lines 77-83 and the cross twiddles at lines 27-30 determine this and must be checked numerically against the direct DFT with a positive exponent.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "Output position k is frequency k in natural order, with the positive exponent omega_N**(j*k); 'This is a cyclic transform, not a negacyclic transform'."
      }
    ],
    "scope_rationale": "The contract requires the forward transform with the POSITIVE exponent omega_N^(j*k) and natural-order output k; problem.txt explicitly rules out negacyclic and conjugated variants.",
    "statement": "The fused radix-4 butterfly twiddle exponents in _make_coefficients (lines 25-30, e.g. omega^(j*width/(2*half)) and the cross products) implement a conjugated or otherwise wrong twiddle mapping, so outputs equal a permuted or inverse-transform (omega^(-j*k)) rather than the requ
  ...[truncated 86 chars]
- `record_claim -> t8 ok`
  Args:
  {
    "rationale": "Tol-0 correctness over the whole domain rests entirely on _multiply being an exact reduction for every reachable (value, coefficient) pair. Source analysis suggests 3327 = -3329^{-1} mod 65536 and bounds are small, but the identity and the single conditional subtract need exhaustive runtime verification; a boundary pair (e.g. product near 3328*3328) could yield reduced >= 2*3329 or a wrong residue.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "EVERY combination of canonical field values 0<=x[b,j]<3329 is legal; implementations may use any internal modular-reduction algorithm 'satisfying this same requirement' (exact, tol 0)."
      }
    ],
    "scope_rationale": "The contract permits any internal modular-reduction algorit
  ...[truncated 533 chars]
  Output summary:
  {
    "id": "c3",
    "rationale": "Tol-0 correctness over the whole domain rests entirely on _multiply being an exact reduction for every reachable (value, coefficient) pair. Source analysis suggests 3327 = -3329^{-1} mod 65536 and bounds are small, but the identity and the single conditional subtract need exhaustive runtime verification; a boundary pair (e.g. product near 3328*3328) could yield reduced >= 2*3329 or a wrong residue.",
    "scope": "in_scope",
    "scope_evidence": [
      {
        "source": "problem.txt",
        "summary": "EVERY combination of canonical field values 0<=x[b,j]<3329 is legal; implementations may use any internal modular-reduction algorithm 'satisfying this same requirement' (exact, tol 0)."
      }
    ],
    "scope_rationale": "The contract permits any internal modular-reduction algorithm but requires exact results (tol 0) for EVERY canonical field-valued input, so _multiply must reduce every reachable product (value in [0,3328] times Montgomery-form coefficient in [0,3328]) to the correct residue in [0,3328].",
    "statement": "The Montgomery-style reduction _multiply (lines 42-45: correction=(product*3327)&65535, reduced=(product+correction*3329)>>16, one conditio
  ...[truncated 167 chars]

### Turn 4 - `experimenter` (21.627433950999993s, in=25583 out=1644)

Message:

```text
I'll launch three independent probes: c1 tests the N=128 path against the reference DFT on GPU, c2 tests N=64 for inverse/permuted-transform signatures, and c3 exhaustively verifies the `_multiply` reduction arithmetic over the full reachable product domain.
```

Tool calls:

- `run_claim_probe -> t9 ok`
  Args:
  {
    "claim_id": "c1",
    "code": "import sys, json\nsys.path.insert(0, \"/root/cases/case_151\")\nimport torch\nfrom kernel import run, reference, make_inputs\n\nresults = {}\nmismatch_total = 0\nfor seed in range(6):\n    x, = make_inputs(device=\"cuda\", batch=16, width=128, seed=seed)\n    out = run(x)\n    ref = reference(x)\n    mism = (out != ref)\n    mismatch_total += int(mism.sum())\n    if seed == 0 and mism.any():\n        idx = mism.nonzero()[0].tolist()\n        b, k = idx\n        results[\"example\"] = {\"row\": b, \"k\": k, \"kernel\": out[b, k].item(), \"ref\": ref[b, k].item()}\n\n# also edge batches: B=1 and B=128, plus all-ones row and boundary values\nx_edge = torch.ones(1, 128, dtype=torch.int32, device=\"cuda\") * 3328\nout = run(x_edge); ref = reference(x_edge)\nedge
  ...[truncated 612 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t9_probe.py",
        "sha256": "4d5ce83075da18a566fae2e2e3fe635bd26d3d6a2b331580b6cc873b7556abe7"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t9_stdout.txt",
        "sha256": "8d528f0be20bd7ab26c4e5245180cf3493e262933a5acf5389df322ca306948d"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t9_stderr.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Parsed JSON object from the last non-empty stdout line.",
        "kind": "json_result",
        "path": "probes/t9_json_result.json",
        "sha256": "f812374f83445ba37a05f0a1bc21191eb0e7f84b8723c57a912ed8222f4dc407"
      }
    ],
    "claim_id": "c1",
    "claim_statement": "For N=128 inputs, the extra odd-LOG_N final radix-2 stage (kernel lines 86-95, coefficient bank from _make_coefficients line 36) produces outputs that differ from the reference modular DFT for some canonical field
  ...[truncated 3322 chars]
- `run_claim_probe -> t10 ok`
  Args:
  {
    "claim_id": "c2",
    "code": "import sys, json\nsys.path.insert(0, \"/root/cases/case_151\")\nimport torch\nfrom kernel import run, reference, make_inputs\n\nQ = 3329\nres = {}\nfor width in (64, 128):\n    x, = make_inputs(device=\"cuda\", batch=4, width=width, seed=3)\n    out = run(x)\n    ref = reference(x)\n    # direct mismatch\n    direct_mism = int((out != ref).sum())\n    # inverse transform signature: out == forward transform with omega^{-j*k}?\n    omega = pow(17, 256 // width, Q)\n    inv = pow(omega, -1, Q)\n    m = torch.tensor([[pow(inv, k * j, Q) for j in range(width)] for k in range(width)],\n                     dtype=torch.int64, device=x.device)\n    ref_inv = (x.to(torch.int64)[:, None, :] * m[None]).sum(2) % Q\n    inv_mism = int((out.to(torch.int64) != ref_inv).su
  ...[truncated 809 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t10_probe.py",
        "sha256": "2d5a0fda469700ae757acc2348fb628a1574e5c0da76e1598fa7dfce493f84ef"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t10_stdout.txt",
        "sha256": "18a36ca1e83ae3b0bc8aad223cc68011249977f2849ba4187b8ff9a2115c8f17"
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
        "sha256": "3a76c9ce5a45144409c85de2af76c614f679bf9c0a8e741b44ac3aef222dd678"
      }
    ],
    "claim_id": "c2",
    "claim_statement": "The fused radix-4 butterfly twiddle exponents in _make_coefficients (lines 25-30, e.g. omega^(j*width/(2*half)) and the cross products) implement a conjugated or otherwise wrong twiddle mapping, so outputs equ
  ...[truncated 4028 chars]
- `run_claim_probe -> t11 ok`
  Args:
  {
    "claim_id": "c3",
    "code": "import sys, json\nimport numpy as np\n\nQ = 3329\nvals = np.arange(0, 3329, dtype=np.int64)\n# Build all reachable coefficient values: every residue can appear as a Montgomery coefficient? Check both the full square range and actual banks.\ndef multiply(v, c):\n    product = v * c  # int64, max 3328*3328 ~ 1.1e7\n    correction = (product * 3327) & 65535\n    reduced = (product + correction * Q) >> 16\n    return reduced - Q if reduced >= Q else reduced\n\n# Exhaustive over full reachable operand grid (value in [0,3328], coefficient in [0,3328])\nV = vals[:, None]\nC = vals[None, :]\nproduct = V * C\ncorrection = (product * 3327) & 65535\nreduced = (product + correction * Q) >> 16\nout = np.where(reduced >= Q, reduced - Q, reduced)\nexpected = (V * C * pow(
  ...[truncated 827 chars]
  Output summary:
  {
    "artifacts": [
      {
        "description": "Python probe code executed by local runtime.",
        "kind": "probe_code",
        "path": "probes/t11_probe.py",
        "sha256": "de613728e65a3627af5910d95078766afc8456321933929fea9116955c678572"
      },
      {
        "description": "Captured stdout from the probe process.",
        "kind": "stdout",
        "path": "probes/t11_stdout.txt",
        "sha256": "c49191a5f659536e5020ee79fe1ae1814a9140e3d221ea9ef3f27930e6e60efb"
      },
      {
        "description": "Captured stderr from the probe process.",
        "kind": "stderr",
        "path": "probes/t11_stderr.txt",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "description": "Parsed JSON object from the last non-empty stdout line.",
        "kind": "json_result",
        "path": "probes/t11_json_result.json",
        "sha256": "ceeb020dc7187aade6c998901e2d5f77e09928e15543f54ac65c56cb5e245d2f"
      }
    ],
    "claim_id": "c3",
    "claim_statement": "The Montgomery-style reduction _multiply (lines 42-45: correction=(product*3327)&65535, reduced=(product+correction*3329)>>16, one conditional subtract) returns a value outside [0,3328] or the wrong residue fo
  ...[truncated 3593 chars]

### Turn 5 - `experimenter` (15.408532676999997s, in=33616 out=928)

Message:

```text
Finalizing all three probe results: c1 confirms the N=128 failure; c2's outputs match neither the forward nor the inverse transform and aren't a permutation, so the specific conjugated/inverse-mapping hypothesis is rebutted; c3's exhaustive sweep found zero wrong reductions, so the Montgomery reducer is rebutted.
```

Tool calls:

- `finalize_probe_evidence -> t12 ok`
  Args:
  {
    "data": {
      "N64_direct_mismatch_observed_in_t10": 256,
      "batch128_mismatch": 16379,
      "edge_all_3328_mismatch": 127,
      "example_mismatch": {
        "k": 0,
        "kernel": 416,
        "ref": 402,
        "row": 0
      },
      "inverse_mismatch_N128": 512,
      "random_mismatch_total": 12283
    },
    "event_id": "t9",
    "status": "confirmed",
    "summary": "N=128 path is broken: massive exact mismatches vs reference DFT across random seeds (12283/12288 elements wrong), all-3328 edge row (127/128 wrong), and full B=128 batch (16379 wrong); example row0 k0 kernel=416 vs ref=402. Note: the c2 probe (t10) also incidentally showed N=64 direct mismatch 256/256, so both supported lengths are wrong, but c1's decisive evidence is the N=128 data.",
    "supports": "confirmed"
  }
  Output summary:
  {
    "claim_id": "c1",
    "claim_status": "confirmed",
    "evidence_id": "c1.e1",
    "summary": "N=128 path is broken: massive exact mismatches vs reference DFT across random seeds (12283/12288 elements wrong), all-3328 edge row (127/128 wrong), and full B=128 batch (16379 wrong); example row0 k0 kernel=416 vs ref=402. Note: the c2 probe (t10) also incidentally showed N=64 direct mismatch 256/256, so both supported lengths are wrong, but c1's decisive evidence is the N=128 data.",
    "supports": "confirmed",
    "tool_event_id": "t9"
  }
- `finalize_probe_evidence -> t13 ok`
  Args:
  {
    "data": {
      "N128": {
        "direct_mismatch": 512,
        "inverse_mismatch": 512,
        "row_multiset_matches_reference": false
      },
      "N64": {
        "direct_mismatch": 256,
        "inverse_mismatch": 256,
        "row_multiset_matches_reference": false
      }
    },
    "event_id": "t10",
    "status": "rebutted",
    "summary": "The specific conjugated/inverse-mapping hypothesis is rebutted: outputs match NEITHER the forward nor the inverse transform (inverse_mismatch == direct_mismatch == all elements at both N=64 and N=128), and row multisets do not match the reference, so the output is not a permutation either. The kernel is wrong, but not via an inverse/negacyclic twiddle mapping as claimed.",
    "supports": "rebutted"
  }
  Output summary:
  {
    "claim_id": "c2",
    "claim_status": "rebutted",
    "evidence_id": "c2.e1",
    "summary": "The specific conjugated/inverse-mapping hypothesis is rebutted: outputs match NEITHER the forward nor the inverse transform (inverse_mismatch == direct_mismatch == all elements at both N=64 and N=128), and row multisets do not match the reference, so the output is not a permutation either. The kernel is wrong, but not via an inverse/negacyclic twiddle mapping as claimed.",
    "supports": "rebutted",
    "tool_event_id": "t10"
  }
- `finalize_probe_evidence -> t14 ok`
  Args:
  {
    "data": {
      "max_reduced_before_subtract": 3494,
      "out_of_range_count": 0,
      "pairs_checked": 11082241,
      "wrong_residue_count": 0
    },
    "event_id": "t11",
    "status": "rebutted",
    "summary": "Exhaustive sweep over all 3329x3329 reachable (value, coefficient) operand pairs: 0 wrong residues, 0 out-of-range results; max reduced before conditional subtract is 3494 < 2*Q, so the single conditional subtract suffices. The Montgomery-style reduction _multiply is exact over the full reachable product domain.",
    "supports": "rebutted"
  }
  Output summary:
  {
    "claim_id": "c3",
    "claim_status": "rebutted",
    "evidence_id": "c3.e1",
    "summary": "Exhaustive sweep over all 3329x3329 reachable (value, coefficient) operand pairs: 0 wrong residues, 0 out-of-range results; max reduced before conditional subtract is 3494 < 2*Q, so the single conditional subtract suffices. The Montgomery-style reduction _multiply is exact over the full reachable product domain.",
    "supports": "rebutted",
    "tool_event_id": "t11"
  }

### Turn 6 - `skeptic` (6.762963671999998s, in=41432 out=239)

Message:

```text
The evidence is decisive: c1 (in_scope, confirmed) shows massive exact mismatches vs the reference DFT on canonical inputs at N=128 (and the N=64 failure is also observed), while the reducer and mapping-mechanism hypotheses were properly resolved. No additional in-scope, testable problem is uncovered that c1 doesn't already cover.
```

Tool calls:

- `record_no_new_claims -> t15 ok`
  Args:
  {
    "reason": "Final review: claim c1 is confirmed, in_scope, and decisive \u2014 run() outputs differ from the exact reference modular DFT (tol 0) on canonical field-valued inputs at N=128 across multiple seeds, edge rows, and full batches (and N=64 failures were also observed in t10). The Montgomery reducer is exhaustively exact (c3 rebutted) and the inverse-mapping mechanism was ruled out (c2 rebutted); neither adds a new defect, and no other untested in-scope failure surface remains material for the verdict.",
    "reviewed_claims": [
      "c1",
      "c2",
      "c3"
    ]
  }
  Output summary:
  {
    "decision": "no_new_claims",
    "reason": "Final review: claim c1 is confirmed, in_scope, and decisive \u2014 run() outputs differ from the exact reference modular DFT (tol 0) on canonical field-valued inputs at N=128 across multiple seeds, edge rows, and full batches (and N=64 failures were also observed in t10). The Montgomery reducer is exhaustively exact (c3 rebutted) and the inverse-mapping mechanism was ruled out (c2 rebutted); neither adds a new defect, and no other untested in-scope failure surface remains material for the verdict.",
    "reviewed_claims": [
      "c1",
      "c2",
      "c3"
    ],
    "reviewed_tool_event_count": 14,
    "turn": 6
  }

### Turn 7 - `judge` (6.943462753999995s, in=42546 out=274)

Message:

```text
Decisive: c1 confirmed, in-scope, tol-0 exact mismatches on canonical inputs. Record reject.
```

Tool calls:

- `record_verdict -> t16 ok`
  Args:
  {
    "confidence": 0.97,
    "decisive_claims": [
      "c1"
    ],
    "reason": "Claim c1 is confirmed, in_scope, and decisive: runtime probes (t9) show run() outputs differ from the exact reference modular DFT on canonical field-valued inputs at N=128 across multiple random seeds (12283/12288 elements wrong), an all-3328 edge row (127/128 wrong), and a full B=128 batch (16379 wrong), e.g. row0 k0 kernel=416 vs ref=402. The contract requires every result to equal the mathematical target exactly (abs/rel tolerance 0) for N in {64,128}; probe t10 additionally shows the N=64 path also mismatches (256/256). The Montgomery reducer itself is exhaustively exact (c3 rebutted) and the specific inverse/conjugated-twiddle mechanism was ruled out (c2 rebutted) \u2014 the kernel is wrong via some other twiddl
  ...[truncated 155 chars]
  Output summary:
  {
    "confidence": 0.97,
    "decisive_claims": [
      "c1"
    ],
    "reason": "Claim c1 is confirmed, in_scope, and decisive: runtime probes (t9) show run() outputs differ from the exact reference modular DFT on canonical field-valued inputs at N=128 across multiple random seeds (12283/12288 elements wrong), an all-3328 edge row (127/128 wrong), and a full B=128 batch (16379 wrong), e.g. row0 k0 kernel=416 vs ref=402. The contract requires every result to equal the mathematical target exactly (abs/rel tolerance 0) for N in {64,128}; probe t10 additionally shows the N=64 path also mismatches (256/256). The Montgomery reducer itself is exhaustively exact (c3 rebutted) and the specific inverse/conjugated-twiddle mechanism was ruled out (c2 rebutted) \u2014 the kernel is wrong via some other twiddle/bank layout error, but the correctness failure against the tol-0 contract is directly established on in-scope canonical inputs.",
    "verdict": "reject"
  }

## Description Model

{
  "contract_model": [
    "Forward cyclic NTT per row: out[b,k] = sum_j x[b,j] * omega_N^(j*k) mod 3329, with omega_N = 17^(256/N) mod 3329, N in {64,128}, positive exponent, natural output order, no normalization.",
    "Exactness required: abs tol 0, rel tol 0, canonical residues in [0,3328].",
    "Domain: ordinary int32 CUDA tensor, shape [B,N] with 1<=B<=128, contiguous (nonzero storage offsets legal), all field values 0..3328 legal; no autograd/inference-mode flag issues; fresh output tensor preserving input.",
    "validate_inputs() enforces the full domain and is called by run(); reference() is a direct modular DFT (int64 products/sums) and accepts CPU.",
    "Single correctness verdict for whole implementation; any transform factorization, internal residue representation, or modular-reduction algorithm is allowed."
  ],
  "kernel_model": [
    "Kernel does a bit-reversal load then radix-4 fused butterfly stages via tl.gather, using precomputed coefficient banks packed in _make_coefficients (lines 19-37), a Montgomery-style reduction in _multiply with R=2^16 (lines 41-45), and an optional final radix-2 stage when LOG_N is odd (lines 86-95).",
    "Coefficients are stored in Montgomery form: (R * omega^step) mod Q; _multiply reduces value*coefficient in Montgomery form, cancelling R since value is plain residue and coefficient carries R.",
    "_multiply: product in int64, correction = (product*3327)&65535, reduced = (product + correction*3329)>>16, then conditional subtract. 3327 = -2 mod 3329... actually 3327 \u2261 -2 mod 3329? 3329-2=3327, so this is a Barrett/Shifu-style reduction whose validity over the reachable product range needs verification (products up to 3328*(R-1)\u22482^32 > int64 is fine, but correction masking works only for product < ~2^31 range claims).",
    "Stage arithmetic keeps residues in [0,3328] after each stage via (value + 2*3329) % 3329 (line 84); quadrant-based fused butterfly mixes sums/differences of Montgomery-multiplied te
...[truncated 4585 chars]

Recent description updates:
- `du1` tasks=`initial`: Initial description of case_151: forward cyclic NTT mod 3329 (omega=17^(256/N), N in {64,128}) implemented as a bit-reversed radix-4 fused-butterfly Triton kernel with Montgomery-form coefficient banks; exact equality (tol 0) over the full canonical field domain.

## Claims

### c1 - `confirmed`

Statement: For N=128 inputs, the extra odd-LOG_N final radix-2 stage (kernel lines 86-95, coefficient bank from _make_coefficients line 36) produces outputs that differ from the reference modular DFT for some canonical field-valued rows.

Scope: `in_scope`

Scope rationale: The contract explicitly requires N=128 as a supported, exactly-correct length ("N is either 64 or 128 ... Every result must equal the mathematical target exactly: absolute tolerance=0 and relative tolerance=0"). The N=128 path (odd LOG_N=7, extra final radix-2 stage at lines 86-95 plus the appended coefficient bank at lines 35-36) must match the reference matrix pow(omega, j*k) exactly.

Scope evidence:
- `problem.txt`: N is either 64 or 128 and every result must equal out[b,k] = sum x[b,j]*omega_N^(j*k) mod 3329 exactly with abs tol 0 and rel tol 0.

Rationale: The odd-LOG_N branch (lines 86-95) is structurally distinct from the fused radix-4 path and is exercised only at N=128: the coefficient start formula `5*((1<<(2*(LOG_N//2)))-1)//3`, the single conditional subtract in _multiply, and the (value+3329)%3329 final reduce all differ from the fused-stage path. Any mismatch in the final bank layout or the bit-reversal interplay at LOG_N=7 would corrupt every N=128 output while N=64 stays correct, and typical make_inputs examples use width=64.

Evidence:
- `c1.e1` runtime_probe supports `confirmed`, tool_event_id=t9: N=128 path is broken: massive exact mismatches vs reference DFT across random seeds (12283/12288 elements wrong), all-3328 edge row (127/128 wrong), and full B=128 batch (16379 wrong); example row0 k0 kernel=416 vs ref=402. Note: the c2 probe (t10) also incidentally showed N=64 direct mismatch 256/256, so both supported lengths are wrong, but c1's decisive evidence is the N=128 data.

### c2 - `rebutted`

Statement: The fused radix-4 butterfly twiddle exponents in _make_coefficients (lines 25-30, e.g. omega^(j*width/(2*half)) and the cross products) implement a conjugated or otherwise wrong twiddle mapping, so outputs equal a permuted or inverse-transform (omega^(-j*k)) rather than the required positive-exponent cyclic transform for some N in {64,128}.

Scope: `in_scope`

Scope rationale: The contract requires the forward transform with the POSITIVE exponent omega_N^(j*k) and natural-order output k; problem.txt explicitly rules out negacyclic and conjugated variants.

Scope evidence:
- `problem.txt`: Output position k is frequency k in natural order, with the positive exponent omega_N**(j*k); 'This is a cyclic transform, not a negacyclic transform'.

Rationale: A twiddle exponent sign error or conjugated mapping is the classic structural bug in NTT kernels: the code compiles, bit-reversal and butterflies look plausible, but the result is the inverse/negacyclic transform. The quadrant-indexed sums/differences at lines 77-83 and the cross twiddles at lines 27-30 determine this and must be checked numerically against the direct DFT with a positive exponent.

Evidence:
- `c2.e1` runtime_probe supports `rebutted`, tool_event_id=t10: The specific conjugated/inverse-mapping hypothesis is rebutted: outputs match NEITHER the forward nor the inverse transform (inverse_mismatch == direct_mismatch == all elements at both N=64 and N=128), and row multisets do not match the reference, so the output is not a permutation either. The kernel is wrong, but not via an inverse/negacyclic twiddle mapping as claimed.

### c3 - `rebutted`

Statement: The Montgomery-style reduction _multiply (lines 42-45: correction=(product*3327)&65535, reduced=(product+correction*3329)>>16, one conditional subtract) returns a value outside [0,3328] or the wrong residue for some reachable product value*coefficient with both operands in [0,3328].

Scope: `in_scope`

Scope rationale: The contract permits any internal modular-reduction algorithm but requires exact results (tol 0) for EVERY canonical field-valued input, so _multiply must reduce every reachable product (value in [0,3328] times Montgomery-form coefficient in [0,3328]) to the correct residue in [0,3328].

Scope evidence:
- `problem.txt`: EVERY combination of canonical field values 0<=x[b,j]<3329 is legal; implementations may use any internal modular-reduction algorithm 'satisfying this same requirement' (exact, tol 0).

Rationale: Tol-0 correctness over the whole domain rests entirely on _multiply being an exact reduction for every reachable (value, coefficient) pair. Source analysis suggests 3327 = -3329^{-1} mod 65536 and bounds are small, but the identity and the single conditional subtract need exhaustive runtime verification; a boundary pair (e.g. product near 3328*3328) could yield reduced >= 2*3329 or a wrong residue.

Evidence:
- `c3.e1` runtime_probe supports `rebutted`, tool_event_id=t11: Exhaustive sweep over all 3329x3329 reachable (value, coefficient) operand pairs: 0 wrong residues, 0 out-of-range results; max reduced before conditional subtract is 3494 < 2*Q, so the single conditional subtract suffices. The Montgomery-style reduction _multiply is exact over the full reachable product domain.

## Tool Events

### t1 - load_artifact - ok

{
  "entry": "case_151"
}

### t2 - inspect_problem - ok

{
  "entry": "case_151"
}

### t3 - list_artifact_files - ok

{
  "entry": "case_151"
}

### t4 - read_artifact_file - ok

{
  "entry": "case_151"
}

### t5 - record_description_update - ok

{
  "contract_model": [
    "Forward cyclic NTT per row: out[b,k] = sum_j x[b,j] * omega_N^(j*k) mod 3329, with omega_N = 17^(256/N) mod 3329, N in {64,128}, positive exponent, natural output order, no normalization.",
    "Exactness required: abs tol 0, rel tol 0, canonical residues in [0,3328].",
    "Domain: ordinary int32 CUDA tensor, shape [B,N] with 1<=B<=128, contiguous (nonzero storage offsets legal), all field values 0..3328 legal; no autograd/inference-mode flag issues; fresh output tensor preserving input.",
    "validate_inputs() enforces the full domain and is called by run(); reference() is a direct modular DFT (int64 products/sums) and accepts CPU.",
    "Single correctness verdict for whole implementation; any transform factorization, internal residue representation, or modular-reduction algorithm is allowed."
  ],
  "impact_on_claims": [],
  "kernel_model": [
    "Kernel does a bit-reversal load then radix-4 fused butterfly stages via tl.gather, using precomputed coefficient banks packed in _make_coefficients (lines 19-37), a Montgomery-style reduction in _multiply with R=2^16 (lines 41-45), and an optional final radix-2 stage when LOG_N is odd (lines 86-95).",
    "Coefficients are stored in Montgomery form: (R * omega^step) mod Q; _multiply reduces value*coefficient in Montgomery form, cancelling R since value is plain residue and coefficient carries R.",
    "_multiply: product in int64, correction = (product*3327)&65535, reduced = (product + correction*3329)>>16, then conditional subtract. 3327 = -2 mod 3329... actually 3327 \u2261 -2 mod 3329? 3329-2=3327, so this is a Barrett/Shifu-style reduction whose validity over the reachable product range needs verification (products up to 3328*(R-1)\u22482^32 > int64 is fine, but correction masking works only for product < ~2^31 range claims).",
    "Stage arithmetic keeps residues in [0,3328] after each stage via (value + 2*3329) % 3329 (line 84); quadrant-based fused butterfly mixes sums/differences o
...[truncated 4955 chars]

### t6 - record_claim - ok

{
  "id": "c1",
  "rationale": "The odd-LOG_N branch (lines 86-95) is structurally distinct from the fused radix-4 path and is exercised only at N=128: the coefficient start formula `5*((1<<(2*(LOG_N//2)))-1)//3`, the single conditional subtract in _multiply, and the (value+3329)%3329 final reduce all differ from the fused-stage path. Any mismatch in the final bank layout or the bit-reversal interplay at LOG_N=7 would corrupt every N=128 output while N=64 stays correct, and typical make_inputs examples use width=64.",
  "scope": "in_scope",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "N is either 64 or 128 and every result must equal out[b,k] = sum x[b,j]*omega_N^(j*k) mod 3329 exactly with abs tol 0 and rel tol 0."
    }
  ],
  "scope_rationale": "The contract explicitly requires N=128 as a supported, exactly-correct length (\"N is either 64 or 128 ... Every result must equal the mathematical target exactly: absolute tolerance=0 and relative tolerance=0\"). The N=128 path (odd LOG_N=7, extra final radix-2 stage at lines 86-95 plus the appended coefficient bank at lines 35-36) must match the reference matrix pow(omega, j*k) exactly.",
  "statement": "For N=128 inputs, the extra odd-LOG_N final radix-2 stage (kernel lines 86-95, coefficient bank from _make_coefficients line 36) produces outputs that differ from the reference modular DFT for some canonical field-valued rows.",
  "status": "open"
}

### t7 - record_claim - ok

{
  "id": "c2",
  "rationale": "A twiddle exponent sign error or conjugated mapping is the classic structural bug in NTT kernels: the code compiles, bit-reversal and butterflies look plausible, but the result is the inverse/negacyclic transform. The quadrant-indexed sums/differences at lines 77-83 and the cross twiddles at lines 27-30 determine this and must be checked numerically against the direct DFT with a positive exponent.",
  "scope": "in_scope",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "Output position k is frequency k in natural order, with the positive exponent omega_N**(j*k); 'This is a cyclic transform, not a negacyclic transform'."
    }
  ],
  "scope_rationale": "The contract requires the forward transform with the POSITIVE exponent omega_N^(j*k) and natural-order output k; problem.txt explicitly rules out negacyclic and conjugated variants.",
  "statement": "The fused radix-4 butterfly twiddle exponents in _make_coefficients (lines 25-30, e.g. omega^(j*width/(2*half)) and the cross products) implement a conjugated or otherwise wrong twiddle mapping, so outputs equal a permuted or inverse-transform (omega^(-j*k)) rather than the required positive-exponent cyclic transform for some N in {64,128}.",
  "status": "open"
}

### t8 - record_claim - ok

{
  "id": "c3",
  "rationale": "Tol-0 correctness over the whole domain rests entirely on _multiply being an exact reduction for every reachable (value, coefficient) pair. Source analysis suggests 3327 = -3329^{-1} mod 65536 and bounds are small, but the identity and the single conditional subtract need exhaustive runtime verification; a boundary pair (e.g. product near 3328*3328) could yield reduced >= 2*3329 or a wrong residue.",
  "scope": "in_scope",
  "scope_evidence": [
    {
      "source": "problem.txt",
      "summary": "EVERY combination of canonical field values 0<=x[b,j]<3329 is legal; implementations may use any internal modular-reduction algorithm 'satisfying this same requirement' (exact, tol 0)."
    }
  ],
  "scope_rationale": "The contract permits any internal modular-reduction algorithm but requires exact results (tol 0) for EVERY canonical field-valued input, so _multiply must reduce every reachable product (value in [0,3328] times Montgomery-form coefficient in [0,3328]) to the correct residue in [0,3328].",
  "statement": "The Montgomery-style reduction _multiply (lines 42-45: correction=(product*3327)&65535, reduced=(product+correction*3329)>>16, one conditional subtract) returns a value outside [0,3328] or the wrong residue for some reachable product value*coefficient with both operands in [0,3328].",
  "status": "open"
}

### t9 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t9_probe.py",
      "sha256": "4d5ce83075da18a566fae2e2e3fe635bd26d3d6a2b331580b6cc873b7556abe7"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t9_stdout.txt",
      "sha256": "8d528f0be20bd7ab26c4e5245180cf3493e262933a5acf5389df322ca306948d"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t9_stderr.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Parsed JSON object from the last non-empty stdout line.",
      "kind": "json_result",
      "path": "probes/t9_json_result.json",
      "sha256": "f812374f83445ba37a05f0a1bc21191eb0e7f84b8723c57a912ed8222f4dc407"
    }
  ],
  "claim_id": "c1",
  "claim_statement": "For N=128 inputs, the extra odd-LOG_N final radix-2 stage (kernel lines 86-95, coefficient bank from _make_coefficients line 36) produces outputs that differ from the reference modular DFT for some canonical field-valued rows.",
  "duration_s": 6.226219,
  "event_id": "t9",
  "evidence_draft": {
    "claim_id": "c1",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t9_probe.py",
          "sha256": "4d5ce83075da18a566fae2e2e3fe635bd26d3d6a2b331580b6cc873b7556abe7"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t9_stdout.txt",
          "sha256": "8d528f0be20bd7ab26c4e5245180cf3493e262933a5acf5389df322ca306948d"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": "stderr",
          "path": "probes/t9_stderr.txt",
          "s
...[truncated 2522 chars]

### t10 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t10_probe.py",
      "sha256": "2d5a0fda469700ae757acc2348fb628a1574e5c0da76e1598fa7dfce493f84ef"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t10_stdout.txt",
      "sha256": "18a36ca1e83ae3b0bc8aad223cc68011249977f2849ba4187b8ff9a2115c8f17"
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
      "sha256": "3a76c9ce5a45144409c85de2af76c614f679bf9c0a8e741b44ac3aef222dd678"
    }
  ],
  "claim_id": "c2",
  "claim_statement": "The fused radix-4 butterfly twiddle exponents in _make_coefficients (lines 25-30, e.g. omega^(j*width/(2*half)) and the cross products) implement a conjugated or otherwise wrong twiddle mapping, so outputs equal a permuted or inverse-transform (omega^(-j*k)) rather than the required positive-exponent cyclic transform for some N in {64,128}.",
  "duration_s": 4.161657,
  "event_id": "t10",
  "evidence_draft": {
    "claim_id": "c2",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t10_probe.py",
          "sha256": "2d5a0fda469700ae757acc2348fb628a1574e5c0da76e1598fa7dfce493f84ef"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t10_stdout.txt",
          "sha256": "18a36ca1e83ae3b0bc8aad223cc68011249977f2849ba4187b8ff9a2115c8f17"
        },
        {
          "description": "C
...[truncated 3228 chars]

### t11 - run_claim_probe - ok

{
  "artifacts": [
    {
      "description": "Python probe code executed by local runtime.",
      "kind": "probe_code",
      "path": "probes/t11_probe.py",
      "sha256": "de613728e65a3627af5910d95078766afc8456321933929fea9116955c678572"
    },
    {
      "description": "Captured stdout from the probe process.",
      "kind": "stdout",
      "path": "probes/t11_stdout.txt",
      "sha256": "c49191a5f659536e5020ee79fe1ae1814a9140e3d221ea9ef3f27930e6e60efb"
    },
    {
      "description": "Captured stderr from the probe process.",
      "kind": "stderr",
      "path": "probes/t11_stderr.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "description": "Parsed JSON object from the last non-empty stdout line.",
      "kind": "json_result",
      "path": "probes/t11_json_result.json",
      "sha256": "ceeb020dc7187aade6c998901e2d5f77e09928e15543f54ac65c56cb5e245d2f"
    }
  ],
  "claim_id": "c3",
  "claim_statement": "The Montgomery-style reduction _multiply (lines 42-45: correction=(product*3327)&65535, reduced=(product+correction*3329)>>16, one conditional subtract) returns a value outside [0,3328] or the wrong residue for some reachable product value*coefficient with both operands in [0,3328].",
  "duration_s": 0.676403,
  "event_id": "t11",
  "evidence_draft": {
    "claim_id": "c3",
    "data": {
      "artifacts": [
        {
          "description": "Python probe code executed by local runtime.",
          "kind": "probe_code",
          "path": "probes/t11_probe.py",
          "sha256": "de613728e65a3627af5910d95078766afc8456321933929fea9116955c678572"
        },
        {
          "description": "Captured stdout from the probe process.",
          "kind": "stdout",
          "path": "probes/t11_stdout.txt",
          "sha256": "c49191a5f659536e5020ee79fe1ae1814a9140e3d221ea9ef3f27930e6e60efb"
        },
        {
          "description": "Captured stderr from the probe process.",
          "kind": 
...[truncated 2793 chars]

### t12 - finalize_probe_evidence - ok

{
  "claim_id": "c1",
  "claim_status": "confirmed",
  "evidence_id": "c1.e1",
  "summary": "N=128 path is broken: massive exact mismatches vs reference DFT across random seeds (12283/12288 elements wrong), all-3328 edge row (127/128 wrong), and full B=128 batch (16379 wrong); example row0 k0 kernel=416 vs ref=402. Note: the c2 probe (t10) also incidentally showed N=64 direct mismatch 256/256, so both supported lengths are wrong, but c1's decisive evidence is the N=128 data.",
  "supports": "confirmed",
  "tool_event_id": "t9"
}

### t13 - finalize_probe_evidence - ok

{
  "claim_id": "c2",
  "claim_status": "rebutted",
  "evidence_id": "c2.e1",
  "summary": "The specific conjugated/inverse-mapping hypothesis is rebutted: outputs match NEITHER the forward nor the inverse transform (inverse_mismatch == direct_mismatch == all elements at both N=64 and N=128), and row multisets do not match the reference, so the output is not a permutation either. The kernel is wrong, but not via an inverse/negacyclic twiddle mapping as claimed.",
  "supports": "rebutted",
  "tool_event_id": "t10"
}

### t14 - finalize_probe_evidence - ok

{
  "claim_id": "c3",
  "claim_status": "rebutted",
  "evidence_id": "c3.e1",
  "summary": "Exhaustive sweep over all 3329x3329 reachable (value, coefficient) operand pairs: 0 wrong residues, 0 out-of-range results; max reduced before conditional subtract is 3494 < 2*Q, so the single conditional subtract suffices. The Montgomery-style reduction _multiply is exact over the full reachable product domain.",
  "supports": "rebutted",
  "tool_event_id": "t11"
}

### t15 - record_no_new_claims - ok

{
  "decision": "no_new_claims",
  "reason": "Final review: claim c1 is confirmed, in_scope, and decisive \u2014 run() outputs differ from the exact reference modular DFT (tol 0) on canonical field-valued inputs at N=128 across multiple seeds, edge rows, and full batches (and N=64 failures were also observed in t10). The Montgomery reducer is exhaustively exact (c3 rebutted) and the inverse-mapping mechanism was ruled out (c2 rebutted); neither adds a new defect, and no other untested in-scope failure surface remains material for the verdict.",
  "reviewed_claims": [
    "c1",
    "c2",
    "c3"
  ],
  "reviewed_tool_event_count": 14,
  "turn": 6
}

### t16 - record_verdict - ok

{
  "confidence": 0.97,
  "decisive_claims": [
    "c1"
  ],
  "reason": "Claim c1 is confirmed, in_scope, and decisive: runtime probes (t9) show run() outputs differ from the exact reference modular DFT on canonical field-valued inputs at N=128 across multiple random seeds (12283/12288 elements wrong), an all-3328 edge row (127/128 wrong), and a full B=128 batch (16379 wrong), e.g. row0 k0 kernel=416 vs ref=402. The contract requires every result to equal the mathematical target exactly (abs/rel tolerance 0) for N in {64,128}; probe t10 additionally shows the N=64 path also mismatches (256/256). The Montgomery reducer itself is exhaustively exact (c3 rebutted) and the specific inverse/conjugated-twiddle mechanism was ruled out (c2 rebutted) \u2014 the kernel is wrong via some other twiddle/bank layout error, but the correctness failure against the tol-0 contract is directly established on in-scope canonical inputs.",
  "verdict": "reject"
}

## Verdict

{
  "confidence": 0.97,
  "created_at": "2026-10-18T03:16:00Z",
  "decisive_claims": [
    "c1"
  ],
  "reason": "Claim c1 is confirmed, in_scope, and decisive: runtime probes (t9) show run() outputs differ from the exact reference modular DFT on canonical field-valued inputs at N=128 across multiple random seeds (12283/12288 elements wrong), an all-3328 edge row (127/128 wrong), and a full B=128 batch (16379 wrong), e.g. row0 k0 kernel=416 vs ref=402. The contract requires every result to equal the mathematical target exactly (abs/rel tolerance 0) for N in {64,128}; probe t10 additionally shows the N=64 path also mismatches (256/256). The Montgomery reducer itself is exhaustively exact (c3 rebutted) and the specific inverse/conjugated-twiddle mechanism was ruled out (c2 rebutted) \u2014 the kernel is wrong via some other twiddle/bank layout error, but the correctness failure against the tol-0 contract is directly established on in-scope canonical inputs.",
  "verdict": "reject"
}
