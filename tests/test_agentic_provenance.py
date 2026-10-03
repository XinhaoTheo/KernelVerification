"""Execution fingerprints follow actual verifier code and prompt skills."""
from verifier.agentic.provenance import verifier_sha256


def test_fingerprint_ignores_location_and_bytecode_but_tracks_code_and_skills(tmp_path):
    roots = [tmp_path / "local/verifier", tmp_path / "remote/verifier"]
    for root in roots:
        (root / "agentic/skills").mkdir(parents=True)
        (root / "agentic/run.py").write_text("return_value = 1\n")
        (root / "agentic/skills/review.md").write_text("Inspect the contract.\n")
    first = verifier_sha256(roots[0])
    assert first == verifier_sha256(roots[1])
    (roots[1] / "compiled.pyc").write_bytes(b"different interpreter cache")
    assert first == verifier_sha256(roots[1])
    (roots[1] / "agentic/skills/review.md").write_text("A changed verification instruction.\n")
    assert first != verifier_sha256(roots[1])
