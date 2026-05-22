# Kernel Verification via Debate

A system that **judges the trustworthiness** of LLM-generated Triton kernels. It does
not trust the generator's own test; instead it independently re-checks correctness and
runs a multi-agent debate, with a focus on catching kernels that "pass their own test
but are actually wrong or gamed."

---

## 1. Why this exists

Tools like meta's KernelAgent auto-generate Triton kernels, but they **set the exam and
sit it at the same time** — the same LLM writes the kernel, writes the test, and often
silently downcasts fp32 to bf16 to loosen tolerance and tests only a single shape. The
result is a kernel that "passes its own test" but may still:

- **Read the wrong memory on non-contiguous inputs** (no `.contiguous()` call)
- **Truncate large inputs** (hard-coded `BLOCK_SIZE`, drops data when a row is too long)
- **Be numerically unstable** (softmax without max-subtraction, drifting reduction order)
- **Cheat** (hard-code the test's shape, fake Triton while calling torch, lazy-eval to fool `allclose`)

This system is the **distrustful, independent reviewer**. It accepts a kernel from any
source (KernelAgent is just the current generator, and is replaceable) and outputs an
evidence-backed trust judgment.

---

## 2. Two-phase architecture

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│  Offline: build dataset  │         │  Online: verify               │
│  (replaceable upstream)  │         │  (our core, self-contained)   │
│                          │  writes │                              │
│  kv-build                │ ──────► │  kv-run <entry>              │
│  KernelAgent gen kernel  │ dataset/│  recheck → debate → verdict  │
└─────────────────────────┘         └──────────────────────────────┘
```

- **Offline (`kv-build`)**: run KernelAgent to turn a problem into a kernel, store it in
  `dataset/`. Expensive, slow, needs a GPU. Run once; the data persists.
- **Online (`kv-run`)**: read a kernel from `dataset/` and run our own verification. Cheap,
  so you can iterate on agent prompts without regenerating kernels.

Key design point: **verification does not depend on the generator providing a test**. Swap
in a generator that emits only `kernel.py` and `kv-run` still works — because it writes its
own test.

---

## 3. End-to-end flow

```
══════════════════════ Offline: kv-build ══════════════════════

 problem (PyTorch reference: Model + get_inputs + get_init_inputs)
   │
   ▼
 verifier/generator.py  ── wraps KernelAgent.TritonKernelAgent
   │   ① LLM writes a test
   │   ② LLM generates N kernel seeds
   │   ③ N workers in parallel: write kernel → subprocess-run test →
   │      feed errors back to the LLM to fix, up to max_rounds
   │   ④ any worker passing = success; on failure also recover the
   │      "best attempt" + its error
   ▼
 verifier/dataset.py : save_entry()
   │   copy kernel/test/seed/problem into a self-contained dir
   ▼
 dataset/<name>/
   ├── problem.txt        original problem
   ├── kernel.py          final kernel (best attempt if it failed)
   ├── test.py            KernelAgent's own test (reference only, not trusted)
   ├── seed_*.py          initial seeds
   ├── meta.json          { passed, status, rounds, ... }
   └── error.txt          stderr/stdout on failure


══════════════════════ Online: kv-run <entry> ══════════════════════

 dataset/<entry>/  ──load_entry()──►  artifact { kernel_code, ... }
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: RECHECK  (verifier/recheck.py — our independent test) │
│                                                               │
│  get_recheck(entry)  (reuses cache; --force-recheck to redo)  │
│    │                                                          │
│    ├─ generate_test():  LLM reads problem+kernel, writes a    │
│    │     test containing a FIXED adversarial battery:         │
│    │       • standard          (spec inputs → core correctness)│
│    │       • noncontig_stride2 (non-contiguous [::2] / [:,::2])│
│    │       • noncontig_transpose (transposed .t())            │
│    │       • odd_size          (size ±1, non-aligned)         │
│    │       • empty             (zero-element tensor)           │
│    │     each case is judged by the fixed compare_outputs()   │
│    │     and prints  "CASE <name>: PASS/FAIL/SKIP"            │
│    │                                                          │
│    ├─ run_test():  temp dir holds kernel.py + kverify_compare │
│    │     .py + test; subprocess runs it; exit code reflects   │
│    │     ONLY the standard case                               │
│    │                                                          │
│    └─ parse CASE lines →  status     = standard's result      │
│                           robustness = the other cases        │
│                                                               │
│  Two tiers (key design):                                      │
│    • standard FAIL  → status=failed → a real bug              │
│    • adversarial FAIL → recorded in robustness, NOT auto-reject│
└─────────────────────────────────────────────────────────────┘
   │  fold recheck result into artifact (passed/status/test_code/error)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: DEBATE  (verifier/debate.py — open-ended review)      │
│                                                               │
│  each round (up to DEBATE_MAX_ROUNDS):                         │
│                                                               │
│    author    ── witness: describes what the kernel does +     │
│       │         how it evolved (describe only, no judgment;   │
│       │         reads seed_*.py)                              │
│       ▼                                                       │
│    skeptic   ── challenger: files structured claims (concrete │
│       │         testable assertions), e.g. {"type":"non_contig│
│       │         ","statement":"for x[::2] the kernel misreads"}│
│       │  ──► registered into the claims ledger, status=open   │
│       ▼                                                       │
│    verifier  ── executor: for each open claim, writes a probe │
│                 that runs on GPU, judges via compare_outputs, │
│                 sets the claim to confirmed / rebutted /      │
│                 inconclusive                                  │
│                                                               │
│    converge: skeptic files no new claim this round → stop    │
│       ▼                                                       │
│    judge     ── renders the final verdict over the ledger:    │
│                   trust / reject / needs_more_evidence        │
│                 + decisive_claims (which claims drove it)    │
│                 + may override verifier (e.g. deems a diff    │
│                   to be expected rounding, not a bug)        │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
 dataset/<entry>/debate_result.json
   { recheck_status, verdict, claims(ledger), history(all turns) }
```

---

## 4. Component deep dive

### 4.1 `verifier/generator.py` — wraps KernelAgent

Normalizes `KernelAgent.TritonKernelAgent.generate_kernel()` into a uniform artifact:
`{kernel_code, test_code, passed, status, rounds, error, session_dir, raw}`.

On success `kernel_code` is the final kernel; on failure it is the "best attempt from the
worker that fought the longest" plus error info (upstream KernelAgent is patched — it
originally left nothing on failure, see [roadmap.md](roadmap.md)).

### 4.2 `verifier/dataset.py` — self-contained dataset

`save_entry()` **copies** each generation's artifacts into `dataset/<name>/` (rather than
recording a path), so deleting KernelAgent's run directory does not break the dataset — it
can be committed, moved between machines, or hand-authored for injection. `load_entry()`
reads it back into an artifact; `session_dir` points at the entry dir itself, which is where
the author reads `seed_*.py` from.

### 4.3 `verifier/recheck.py` — independent correctness re-check (our ground truth)

This is the heart of **distrusting the generator**. `get_recheck()`:

1. **`generate_test()`**: ask the LLM to read `problem.txt` + `kernel.py` and write a test.
   Since the LLM sees the kernel source, it knows how to call it (this sidesteps the "every
   kernel has a different call signature" problem). The prompt forces it to run a **fixed
   adversarial battery**, each case printing `CASE <name>: PASS/FAIL/SKIP`.
2. **`run_test()`**: a temp dir holds `kernel.py` + `kverify_compare.py` (the fixed
   comparator) + the test; run it in a subprocess. A fresh process, avoiding the CUDA fork
   trap.
3. **Parse the `CASE` lines** into two tiers stored in `meta.json["recheck"]`:
   - `status`: from `standard` (spec inputs) only → core correctness, this is what drives
     "is it a real bug"
   - `robustness`: the other adversarial cases → recorded only, **not auto-reject**

**Why the battery**: previously only the debate skeptic probed adversarial inputs ad hoc,
which is non-deterministic — the same kernel with a non-contiguous bug would be caught one
run (skeptic thought of it) and missed the next (it didn't). The battery hard-codes the list
and runs it every time, so mechanical bugs (non-contiguous / odd size / empty) are no longer
missed.

**What the two tiers mean (the scope contract)**: a kernel that fails on the spec's normal
inputs is an unambiguous bug (reject); one that only fails on adversarial inputs like
non-contiguous is a robustness gap — recorded but not condemned, because such inputs may be
outside the kernel's intended scope.

### 4.4 `verifier/compare.py` — the single source of truth for comparison

`compare_outputs(out, ref) → (matches, max_diff, detail)`. Both the recheck test and the
verifier probe **import it** (copied into the temp dir as `kverify_compare.py` at run time),
so the "is it correct" decision lives in exactly one place and is fixed:

- **Tolerance by dtype**: fp32 uses 1e-3, fp16/bf16 uses 1e-2/2e-2 (the LLM is not allowed to
  invent its own threshold like `1e-4`)
- **`equal_nan=True`**: kernel and reference both NaN at the same position counts as a match
  (softmax of all-inf yields NaN in PyTorch too — that is not the kernel's fault)
- **A bug means "diverges from the reference"**, never "produced a NaN/large value" in isolation

> This file was added after a bug: early on the LLM wrote the comparison inside each probe,
> picked tolerances arbitrarily, and mishandled NaN, producing false positives. Extracting a
> fixed comparator made recheck and verifier use the same yardstick.

### 4.5 `verifier/debate.py` + `agents/` — multi-agent debate

Debate handles the **semantic / algorithmic bugs the battery cannot enumerate** (e.g. cumsum
cross-block accumulation errors, cheating, subtle numerics). Four roles:

- **`agents/author.py` (witness)**: reads the final kernel + `seed_*.py` and objectively
  describes what the kernel does and what changed from seed to final. The prompt forbids
  quality judgment — describe only. Emits `NO_NEW_OBSERVATIONS.` when it has nothing to add.
- **`agents/skeptic.py` (challenger)**: looks for possible bugs/cheating, but must file them
  as **structured claims** — a concrete assertion verifiable by running code (with the exact
  input to test). Emits `NO_NEW_CONCERNS.` + an empty claim list when out of new concerns.
- **`agents/verifier.py` (executor)**: walks the `open` claims in the ledger, **writes a probe
  per claim** that constructs the input, runs kernel + reference, judges via `compare_outputs`,
  and writes the result back to the claim (confirmed / rebutted / inconclusive) plus evidence
  (probe code + measured numbers).
- **`agents/judge.py` (arbiter)**: does not speak per round — it **reads the whole ledger once
  at the end** and renders the verdict. It makes the severity call counting cannot: is a
  confirmed diff of 256 a real bug or expected bf16 rounding? The judge may **override** the
  verifier's call. Outputs `{verdict, confidence, decisive_claims, claim_notes, reason}`.

- **`agents/parsing.py`**: robustly extracts JSON from an LLM reply (prefers the ```json block,
  skips ```python code blocks that appear in prose).
- **`agents/types.py`**: the `Turn` / `Claim` TypedDict definitions.

**Convergence**: each round runs author → skeptic → verifier; it stops when the skeptic files
no new claim. The judge is not part of the rounds; it adjudicates once after the loop.

### 4.6 `verifier/llm_client.py` — LLM calls

A single Anthropic wrapper. Two points: (1) the API is stateless, so we own the message list;
(2) the message layout lets the prompt cache hit across rounds for the same agent (kernel/test
go in the cacheable prefix). `oneshot()` is the historyless call used by recheck/verifier to
write probes.

### 4.7 `verifier/gpu_pick.py` — automatic GPU selection

Before running, sort GPUs via `nvidia-smi` and then actually attempt a small allocation as a
probe (a card can report "free" in nvidia-smi yet refuse a context on a shared machine), then
set `CUDA_VISIBLE_DEVICES` to the first usable one.

---

## 5. The three signals

After `kv-run` finishes a kernel, you get three layers of judgment — **do not conflate them**:

| Signal | Source | Meaning |
|---|---|---|
| `recheck.status` | the battery's `standard` case | **Core correctness**: right on the spec's normal inputs? |
| `recheck.robustness` | the battery's adversarial cases | **Robustness**: handles non-contiguous / odd size / empty? |
| `debate.verdict` | author/skeptic/verifier/judge | **Semantic review**: algorithmic bugs / cheating the battery can't enumerate |

> Note: these three signals are currently produced separately; how to combine them into one
> final conclusion is still an open design point.

---

## 6. Repository layout

```
kernel_verification/
├── KernelAgent/        # upstream generator (meta-pytorch/KernelAgent), patched, vendored
├── KernelBench/        # problem set (ScalingIntelligence/KernelBench), vendored
├── verifier/           # main package
│   ├── generator.py    # wraps KernelAgent
│   ├── dataset.py      # save_entry / load_entry / iter_entries
│   ├── recheck.py      # independent re-check + adversarial battery (kv-recheck)
│   ├── compare.py      # fixed comparator compare_outputs (single source of truth)
│   ├── debate.py       # debate main loop run_debate
│   ├── llm_client.py   # Anthropic calls + prompt cache
│   ├── gpu_pick.py     # auto-pick a usable GPU
│   ├── build_dataset.py# offline dataset build (kv-build)
│   └── run.py          # online verification entry (kv-run)
├── agents/             # the four roles + helpers
│   ├── author.py  skeptic.py  verifier.py  judge.py
│   ├── parsing.py      # JSON extraction
│   └── types.py        # Turn / Claim
├── dataset/            # generated kernels + three labels (committable)
│   └── <name>/{problem.txt, kernel.py, test.py, seed_*.py,
│               meta.json, recheck_test.py, debate_result.json, ...}
├── tests/              # exploration / debugging scripts
├── pyproject.toml      # uv project, torch cu128, KernelAgent path dep
└── roadmap.md          # design evolution notes
```

---

## 7. CLI reference

```bash
# Offline: build the dataset (expensive/slow/needs GPU, run once)
uv run kv-build --curated              # build 10 curated KernelBench problems
uv run kv-build --problem elem_add     # build a single built-in problem
uv run kv-build --curated --list       # dry run, just show what would be built

# Independent re-check (only recheck, no debate)
uv run kv-recheck                       # run everything not yet rechecked
uv run kv-recheck elem_add              # force re-run one
uv run kv-recheck --list                # show each entry's recheck status

# Online: verify (recheck → debate → verdict)
uv run kv-run elem_add                  # run one
uv run kv-run elem_add --verbose        # watch it all: recheck test / each agent turn / probe code
uv run kv-run elem_add --force-recheck  # regenerate the recheck test, then run
uv run kv-run --list                    # list runnable entries
```

After a run, see `dataset/<entry>/debate_result.json` for the full record.

---

## 8. Known limitations

- **The debate verdict is still non-deterministic**: the battery made mechanical bugs
  (non-contiguous, etc.) deterministic, but inside the debate the skeptic still files claims
  ad hoc, so coverage of semantic bugs still varies run-to-run.
- **Battery cases are still LLM-written**: the checklist is fixed, but each case's concrete
  construction is still generated by the LLM, so there is construction variance. Eliminating
  it fully requires a pure-Python harness (which hits the "call any kernel generically"
  signature problem).
- **The three signals are not combined**: `recheck.status` / `robustness` / `debate.verdict`
  are produced separately; how to synthesize a final conclusion is undecided.
- **The scope contract is not fixed**: whether inputs like non-contiguous "count as a bug"
  depends on the kernel's intended use; the current compromise is "fail on normal inputs =
  condemned, fail on adversarial inputs = recorded only."
