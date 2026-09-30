"""Offline regression coverage for lossless benchmark trace storage."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from benchmark_fn_fp.eval_scripts import traces


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    root.mkdir()
    monkeypatch.setattr(traces, "BENCHMARK_DIR", root)
    return root


def archive(members):
    """Construct controlled archives, including unsafe members for rejection."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as handle:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "../../outside"
            info.size = len(content) if kind == tarfile.REGTYPE else 0
            handle.addfile(info, io.BytesIO(content) if info.isfile() else None)
    return output.getvalue()


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_archive_roundtrip_preserves_binary_probes_and_raw_calls(benchmark, tmp_path):
    source = tmp_path / "source_run"
    payloads = {
        "run.json": b'{"verdict":{"verdict":"trust"}}',
        "transcript.md": "# Transcript\n\n误差已验证\n".encode(),
        "tool_events.jsonl": b'{"tool":"run_probe"}\n',
        "probes/probe_1/main.py": b"print('probe')\n",
        "probes/probe_1/output.bin": bytes(range(256)) + b"\x00\xff",
        "llm_calls/0001/request.json": b'{"messages":[{"content":"Inspect"}]}',
        "llm_calls/0001/response.json": b'{"reasoning_content":"raw","finish_reason":"length"}',
    }
    for name, data in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    dest = traces.reserve_trace("case_a", "solo", traces_dir="traces_glm", trial="roundtrip",
        metadata={"model": "accounts/fireworks/models/glm-5p3", "provider": "fireworks"})
    result = traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="roundtrip",
        tar=traces.pack_run_dir(source), metadata={"status": "completed"})
    assert result == dest
    assert {name: (dest / name).read_bytes() for name in payloads} == payloads
    meta = json.loads((dest / "trace_meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["provider"] == "fireworks"
    assert meta["model"] == "accounts/fireworks/models/glm-5p3"
    # Importing an identical payload again is safe and preserves bytes.
    before = snapshot(dest)
    traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="roundtrip",
        tar=traces.pack_run_dir(source))
    assert snapshot(dest) == before


def test_conflicting_file_rejected_before_any_payload_write(benchmark):
    dest = traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="existing",
        files={"run.json": "original"})
    before = snapshot(dest)
    with pytest.raises(FileExistsError, match="overwrite"):
        traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="existing",
            files={"new.txt": "must not appear", "run.json": "replacement"})
    assert snapshot(dest) == before


def test_file_parent_conflict_rejected_before_any_payload_write(benchmark):
    dest = traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="parent_conflict",
        files={"probes": "existing regular file"})
    before = snapshot(dest)
    with pytest.raises((OSError, ValueError)):
        traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="parent_conflict",
            files={"new.txt": "must not appear", "probes/main.py": "print(1)"})
    assert snapshot(dest) == before


@pytest.mark.parametrize("name,kind", [
    ("../outside.txt", tarfile.REGTYPE),
    ("/absolute/outside.txt", tarfile.REGTYPE),
    ("nested/../../outside.txt", tarfile.REGTYPE),
    ("link", tarfile.SYMTYPE),
    ("hard_link", tarfile.LNKTYPE),
    ("pipe", tarfile.FIFOTYPE),
])
def test_unsafe_archive_rejected_before_any_payload_write(benchmark, name, kind):
    data = archive([("good.txt", b"must not appear", tarfile.REGTYPE), (name, b"bad", kind)])
    with pytest.raises(ValueError, match="Unsafe|Non-regular"):
        traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial="unsafe", tar=data)
    assert list(benchmark.iterdir()) == []


def test_shared_tree_trials_are_exclusive_even_across_providers(benchmark):
    destinations = []
    for provider, trial in (("openrouter", "r1"), ("fireworks", "r2")):
        dest = traces.reserve_trace("case_a", "solo", traces_dir="traces_glm",
            trial=trial, metadata={"provider": provider})
        destinations.append(dest)
        traces.write_trace("case_a", "solo", traces_dir="traces_glm", trial=trial,
            files={"run.json": f"{provider}:{trial}"})
        with pytest.raises(FileExistsError):
            traces.reserve_trace("case_a", "solo", traces_dir="traces_glm", trial=trial,
                metadata={"provider": "different_provider"})
        assert (dest / "run.json").read_text() == f"{provider}:{trial}"
    assert len(set(destinations)) == 2


def test_discovery_preserves_legacy_and_ignores_nested_payloads(benchmark):
    expected = {}
    for root, provider, model in (
        ("traces_glm", "openrouter", "z-ai/glm-5.3-flash"),
        ("traces_glm_fireworks", "fireworks", "accounts/fireworks/models/glm-5p3"),
        ("traces_opus5", "anthropic", "claude-opus-5"),
    ):
        legacy = benchmark / root / "case_01" / "solo"
        legacy.mkdir(parents=True)
        (legacy / "run.json").write_text("{}")
        (legacy / "usage.json").write_text("{}")
        expected[legacy] = ("legacy", provider, model)
        # A legacy flat arm's nested diagnostic usage files are not trials.
        for nested in ("llm_calls", "llm_calls/call_1", "probes", "probes/probe_1"):
            child = legacy / nested
            child.mkdir(parents=True, exist_ok=True)
            (child / "usage.json").write_text("{}")
    for provider, model, trials in (("fireworks", "accounts/fireworks/models/glm-5p3", ("r1", "r2")),
                                    ("openrouter", "z-ai/glm-5.3-flash", ("r3", "r4"))):
        for trial in trials:
            dest = traces.reserve_trace("case_a", "single_call", traces_dir="traces_glm",
                trial=trial, metadata={"provider": provider, "model": model, "dataset": "correlation_pair"})
            (dest / "usage.json").write_text("{}")
            for nested in ("llm_calls/call_1", "probes/probe_1"):
                child = dest / nested
                child.mkdir(parents=True)
                (child / "usage.json").write_text("{}")
            expected[dest] = (trial, provider, model)
    records = list(traces.iter_trace_records())
    assert {row["path"] for row in records} == set(expected)
    assert len(records) == len(expected)
    for row in records:
        assert (row["trial"], row["provider"], row["model"]) == expected[row["path"]]
