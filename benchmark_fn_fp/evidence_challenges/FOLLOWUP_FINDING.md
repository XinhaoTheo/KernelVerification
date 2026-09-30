# Follow-up experiment scheduling: finding from case_75

Observed in frozen `ea_methods_v3_r1` and `ea_methods_v3_r2`; no runtime or prompt
change was made during either cohort. This is an implementation proposal, not a
tested fix and not a replacement score for the two abstentions.

In r1 the Experimenter detected an invalid numerical reference, marked the
claims inconclusive and promised a repaired experiment. The follow-up did not
run: experiment coverage is driven by open claims, and these claims were now
inconclusive. The Judge's reject attempt was correctly blocked by the ledger.
In r2 a repaired experiment produced evidence of failure, but the decisive
claim was not finalized with that evidence. Both final outcomes were abstention.

Relevant existing code: [orchestrator](../../verifier/agentic/orchestrator.py),
[Experimenter instructions](../../verifier/agentic/agents/experimenter.py),
[verdict validation](../../verifier/agentic/tools/verdict.py).
Full experimental evidence is in [the precision audit](METHODS_V3_AUDIT_PRECISION.md).

A future version can add a structured follow-up request containing claim IDs,
the invalid observation or missing finalization, and the next concrete action.
While the original run budget remains, schedule a bounded Experimenter repair
turn before the final Judge. Existing inconclusive claims must be eligible when
they have a specific pending repair; retrying every inconclusive claim blindly
would waste budget. After a successful probe, check that the requested evidence
has been attached and the claim updated, or retain a stated unresolved blocker.
Do not auto-confirm a claim merely because a probe executed successfully.

Preserve the ledger's evidence requirements. If the repair is impossible or the
original budget is spent, abstention is appropriate. A future regression should
exercise invalid reference → follow-up request → repaired probe → explicit
claim finalization → Judge, plus the unrecoverable and exhausted-budget paths.
Compare that workflow in a separate version on fresh cases; do not silently
change the baseline in the ongoing transfer experiment.
