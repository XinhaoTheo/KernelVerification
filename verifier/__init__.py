"""Kernel verification package."""

def __getattr__(name):
    if name == "generate_kernel":
        from .generator import generate_kernel

        return generate_kernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
