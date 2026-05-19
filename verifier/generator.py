"""Wrapper around meta-pytorch/KernelAgent's TritonKernelAgent.

Normalizes the upstream return shape into the form our debate layer expects:
    {kernel_code, test_code, passed, session_dir, raw}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triton_kernel_agent import TritonKernelAgent


def generate_kernel(
    problem_description: str,
    *,
    agent: TritonKernelAgent | None = None,
    test_code: str | None = None,
    generate_default_test: bool = True,
) -> dict[str, Any]:
    """Run KernelAgent on a problem and normalize the result.

    Args:
        problem_description: PyTorch reference impl + shape/dtype spec, as a string.
            See KernelAgent/examples/triton_01_element_add.py for the canonical
            "Model + get_inputs + get_init_inputs" layout.
        agent: Optional pre-built TritonKernelAgent. Pass one to reuse provider /
            worker pool across many generate_kernel calls.
        test_code: Extra correctness test source to append after the LLM-generated
            one. Must `from kernel import kernel_function` and sys.exit(0/1).
        generate_default_test: If False, you MUST provide test_code.

    Returns:
        {
            "kernel_code": str,    # final kernel on pass; best failed attempt on fail; "" if neither
            "test_code":   str,    # contents of session_dir/test_0.py (or test_1.py if 0 missing)
            "passed":      bool,   # mirrors upstream "success"
            "status":      str,    # "passed" | "compile_error" | "test_failed" | "stopped_early"
                                   # | "no_kernel_generated" | "unknown_failure"
            "rounds":      int|None,
            "error": {             # only meaningful when passed=False
                "stderr": str,
                "stdout": str,
            },
            "session_dir": str,    # always present; evidence pool for the skeptic
            "raw":         dict,   # upstream return dict, untouched
        }
    """
    owns_agent = agent is None
    if owns_agent:
        agent = TritonKernelAgent()

    try:
        result = agent.generate_kernel(
            problem_description,
            test_code=test_code,
            generate_default_test=generate_default_test,
        )
    finally:
        if owns_agent:
            agent.cleanup()

    session_dir = Path(result["session_dir"])
    test_code_str = _read_first_test(session_dir)
    passed = bool(result.get("success", False))

    status = "passed" if passed else result.get("status", "unknown_failure")
    rounds = result.get("rounds")

    return {
        "kernel_code": result.get("kernel_code", ""),
        "test_code": test_code_str,
        "passed": passed,
        "status": status,
        "rounds": rounds,
        "error": {
            "stderr": result.get("last_stderr", ""),
            "stdout": result.get("last_stdout", ""),
        },
        "session_dir": str(session_dir),
        "raw": result,
    }


def _read_first_test(session_dir: Path) -> str:
    """Return the first test_*.py file's contents, or '' if none exists."""
    for candidate in sorted(session_dir.glob("test_*.py")):
        return candidate.read_text()
    return ""
