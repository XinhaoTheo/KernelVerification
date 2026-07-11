from __future__ import annotations

import json
from pathlib import Path

from verifier.agentic_run import main as agentic_main


def test_agentic_cli_all_dry_run_writes_each_entry(tmp_path, capsys) -> None:
    _write_artifact(tmp_path / "dataset", "toy1")
    _write_artifact(tmp_path / "dataset", "toy2")

    exit_code = agentic_main(
        [
            "--all",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "entry: toy1" in captured.out
    assert "entry: toy2" in captured.out
    assert (tmp_path / "runs" / "toy1" / "run.json").exists()
    assert (tmp_path / "runs" / "toy2" / "run.json").exists()


def test_agentic_cli_openai_provider_uses_factory(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset", "toy")
    calls = []

    class FakeLLMClient:
        def call(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
            return json.dumps({"message": "No claims.", "tool_calls": []})

    def fake_build_llm_client(*, provider=None, model=None):
        calls.append({"provider": provider, "model": model})
        return FakeLLMClient()

    monkeypatch.setattr("verifier.agentic_run.build_llm_client", fake_build_llm_client)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--provider",
            "openai",
            "--model",
            "gpt-5",
            "--agent",
            "describer",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [{"provider": "openai", "model": "gpt-5"}]
    assert "provider: openai" in captured.out


def _write_artifact(dataset_root: Path, name: str) -> None:
    entry_dir = dataset_root / name
    entry_dir.mkdir(parents=True)
    (entry_dir / "meta.json").write_text(
        json.dumps({"name": name, "passed": True, "status": "passed", "rounds": 1})
    )
    (entry_dir / "problem.txt").write_text("Add one to every element.\n")
    (entry_dir / "kernel.py").write_text("def kernel(x):\n    return x + 1\n")
    (entry_dir / "test.py").write_text("def test():\n    pass\n")
