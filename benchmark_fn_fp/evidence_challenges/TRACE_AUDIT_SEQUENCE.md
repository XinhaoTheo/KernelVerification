# case_64 / case_65 pilot trace audit

Storage update (2026-09-30): Batch labels and repeat shorthand below retain their original experimental meaning; original labels are saved in `trace_meta.json.original_trial`. Canonical links use per-case/arm `rN` folders; see [trace naming and migration](../TRACES.md).

Audit date: 2026-09-24. Trial: `ea_pilot_r1`. Scope: the four completed
`solo` / `debate` runs for the sequence-audit pair. This audit only reads
saved artifacts and performs local hash/JSON checks; it does not rerun GPU or
model calls, change cases, or select replacement examples.

## Result

Both tool arms produce the correct verdict for both cases. This pair provides
**no observed debate accuracy advantage over solo**. Solo independently noticed
the incomplete smoke-test coverage and used one exhaustive GPU probe per case.
Debate added targeted probes and a more explicit permutation explanation, but
its final classification was the same. The experimenter also introduced and
repaired two failed reference-code probes for case_64.

| Case | Frozen truth | Solo evidence | Debate evidence | Verdicts |
|---|---|---|---|---|
| case_64 | trust | All 121 words, 0 failures, measured maximum error 0 | All 121 words, 0 errors above tolerance; targeted word exact | trust / trust |
| case_65 | reject | All 121 words, 12 failures, maximum error 7.625 | All 121 words, 12 failures, maximum error 7.625; targeted counterexamples | reject / reject |

The contract is the fixed seeded cache and increment with all words of length
0–4 over operations `{0,1,2}`, including the empty word. Each word starts from
a fresh cache. Reorders operate on current logical rows and each append adds
the same increment to current logical rows. The tolerance is maximum absolute
error `1e-5`; output shape, dtype and finiteness are also required.

## case_64: solo

[Probe t7](../traces_glm/case_64/solo/r1/probes/t7_probe.py)
enumerates `range(5)` and `itertools.product((0,1,2), repeat=L)`, asserts 121
words, calls the actual `kernel.run_sequence` with CUDA inputs, and evaluates
an independent forward FP64 logical reference. The reference starts from
`C.copy()` for every word; `run_sequence` itself clones the initial cache.
The probe checks `(8,32)`, float32, finite outputs and the contractual metric.
[Its output](../traces_glm/case_64/solo/r1/probes/t7_stdout.txt)
reports all 121 words, 0 failures, and maximum error 0.0. The
[trust verdict](../traces_glm/case_64/solo/r1/verdict.json) is supported.

The initial reasoning incorrectly suggested that any nonidentity order might
fail. The same solo agent rebutted this hypothesis after its probe. Its final
phrase “internally consistent” must be read within this fixed-input contract:
the implementation is not correct for arbitrary reorder pairs. The private
oracle shows that this case's two involutions commute, so every reachable
order is an involution. A missing `scope_rationale` in the first claim-tool
request was repaired; no GPU-probe failure occurred.

## case_64: debate

The Describer recognized the logical-to-physical mapping, but its risk map and
the Skeptic initially overgeneralized that mixed reorders should fail. In
particular, the conjugate word `(0,1,0,2)` need not fail: a conjugate of an
involution is an involution. The Experimenter did not treat these hypotheses
as established verdicts.

Initial probes
[t12](../traces_glm/case_64/debate/r1/probes/t12_probe.py) and
[t13](../traces_glm/case_64/debate/r1/probes/t13_probe.py) incorrectly
used `C[0]` after unpacking the initial cache and crashed with broadcasting
errors. Their stderr and source are preserved. The Experimenter identified
this error and reran corrected probes; it did not claim either failed probe
was evidence for the kernel label.

Corrected [t14](../traces_glm/case_64/debate/r1/probes/t14_probe.py)
evaluates `(0,1,2)` against the proper FP64 logical reference and obtains
error 0. It prints the two actual seeded permutations and an involutive
composition. Its diagnostic calls `P[1][P[0]]` the order after `(0,1)`;
the implementation's forward order is `P[0][P[1]]`. For this particular
commuting pair they are identical, so this naming/direction error does not
affect the measured reference result or the diagnosis here.

Corrected [t15](../traces_glm/case_64/debate/r1/probes/t15_probe.py)
does run all 121 legal words on the GPU and uses the correct FP64 reference.
[Its output](../traces_glm/case_64/debate/r1/probes/t15_stdout.txt)
reports no above-tolerance errors. However, two reporting limitations matter:

- It accumulates only entries with `err > 1e-5` and computes
  `max_err_overall` over those failing entries. A reported zero therefore
  means “no recorded above-tolerance errors”, not an independently measured
  zero maximum over every result.
- It does not explicitly check shape, dtype or finiteness. In particular,
  a NaN metric would not enter the failure list. The inspected source
  creates the correct output shape/dtype and performs finite, exactly
  representable bounded arithmetic on these fixed inputs; the frozen oracle
  and solo trace separately verify all conditions. Those facts must not be
  misattributed to t15's checks.

The [trust verdict](../traces_glm/case_64/debate/r1/verdict.json)
is correct. Its claim that t15 directly measured maximum error exactly zero
is stronger than that probe's reporting establishes. The final Skeptic review
did not flag either of these probe limitations. Thus this is not evidence of
a uniquely effective multi-role evidence audit.

## case_65: solo

[Probe t6](../traces_glm/case_65/solo/r1/probes/t6_probe.py)
independently sweeps all 121 words with the correct FP64 logical reference,
fresh initial state, GPU kernel execution, and shape/dtype/finite/metric checks.
[Its output](../traces_glm/case_65/solo/r1/probes/t6_stdout.txt)
matches the frozen oracle: 12 failing words and maximum absolute error 7.625.

Concrete legal counterexamples are `(0,1,2)` and `(1,0,2)`, each with error
3.8125, and `(0,1,2,2)` with error 7.625. Any of these already establishes
rejection. The [reject verdict](../traces_glm/case_65/solo/r1/verdict.json)
correctly treats the initial five smoke successes as incomplete coverage.
The phrase “non-identity reorder composition” should not be read as a
sufficient failure condition: non-involutive order is the relevant direction
mismatch, and several nonidentity involutions are harmless here.

## case_65: debate

[Targeted t12](../traces_glm/case_65/debate/r1/probes/t12_probe.py)
checks eight legal words using the independent FP64 logical reference.
[Its output](../traces_glm/case_65/debate/r1/probes/t12_stdout.txt)
shows the seeded composition is not an involution, the two 3.8125-error words,
the repeated-append 7.625-error word, and five passing controls. This is valid
counterexample evidence. Targeted t12 does not check every output condition,
but a numeric error above tolerance alone suffices to reject.

[Exhaustive t13](../traces_glm/case_65/debate/r1/probes/t13_probe.py)
additionally checks the full domain and all output conditions.
[Its output](../traces_glm/case_65/debate/r1/probes/t13_stdout.txt)
matches solo and the private oracle: 121 words, 12 failures, maximum error
7.625. The [reject verdict](../traces_glm/case_65/debate/r1/verdict.json)
is supported by both probes.

The Skeptic improved the Describer's initial indexing explanation by stating
that physical slot `p` must receive `Delta[order_inverse[p]]`. Nevertheless,
some intermediate claims were too strong: exhaustive enumeration is not
necessary to reject once a valid counterexample exists, and repeated appends
do not themselves change the permutation's involution property. Targeted
controls disprove some initially suggested failing words. The final verdict
uses the valid measured counterexamples, so these speculative inaccuracies
do not invalidate it. No correction unique to debate was needed to obtain
the final label; solo found the same failures directly.

## Trace integrity

All four runs have completed metadata. Their public source and contract hashes
match the current frozen files, and their `run.json` source/contract snapshots
match byte-for-byte. The model is `accounts/fireworks/models/glm-5p3`, reasoning
effort `low`, with a 32768 total output-token budget per run.

| Run | Raw API requests/responses/metadata | Output tokens | Saved probe attempts | Verified tool artifact hashes |
|---|---:|---:|---:|---:|
| case_64 solo | 5 / 5 / 5 | 2675 | 1 | 4 |
| case_64 debate | 10 / 10 / 10 | 7042 | 4, including 2 failures | 14 |
| case_65 solo | 4 / 4 / 4 | 1317 | 1 | 4 |
| case_65 debate | 9 / 9 / 9 | 7357 | 2 | 8 |

All 28 captured calls have completed metadata, saved responses and expected
model/reasoning settings. All 30 saved probe artifacts referenced by tool
events have matching SHA-256 hashes, including the failed case_64 probe artifacts.
No additional model calls, GPU executions or case modifications were made for
this audit.

Probe source SHA-256 values:

| Run / probe | SHA-256 |
|---|---|
| case_64 solo t7 | `cb44bb630e0b8737a7b3726ae5dbe21bbffaa77bbc9215e862fba0e127d27402` |
| case_64 debate t12, failed | `8f8aca2fc008cfd930c043bf8607542458deaee0fa20e769c5131498bad8c25e` |
| case_64 debate t13, failed | `2e9b0b7fb1334792f5665bd64953ee6203f900307c247636ab9bde610b47e7ae` |
| case_64 debate t14 | `6f7e6ca8e55b644e1df325e30048374b0e6015038ef976ba7a2a7b1ee807a687` |
| case_64 debate t15 | `23be338bc7aaa98be3eadfdc3359d3c0fa9028e9ef6b62864b3829fe87a2bb3f` |
| case_65 solo t6 | `f5f225695805b550142a69f22ae11ff28d226274fef5ae3f5d1f0d3793e9a176` |
| case_65 debate t12 | `dbcab87ae55a205f69d3c5d5503661da49d84247be5e98d1028d01d66ee9e071` |
| case_65 debate t13 | `9fb24c230da374ed4ac70ade3a230599918ad8d009fece79ce931811c54155fc` |
