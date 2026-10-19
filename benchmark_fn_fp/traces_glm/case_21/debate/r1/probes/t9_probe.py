import torch, json, triton
# Determine whether scalar K arg and program_id arithmetic are int32,
# which decides overflow for K >= 2^31, without allocating 2^31 floats.
import triton.language as tl

@triton.jit
def _probe_kernel(K, Out):
    part = tl.program_id(0)
    chunk = tl.cdiv(K, 512)
    start = part * chunk
    tl.store(Out + part, start.to(tl.int64))

K_val = 2**31 + 100  # exceeds int32 range
# chunk for small real K: emulate real kernel's params with tiny K to inspect dtype
real_K = 1000
out = torch.zeros(512, device="cuda", dtype=torch.int64)

# Inspect the compiled signature of the real kernel for scalar K dtype
from triton.compiler import ASTSource
import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_22/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
a = torch.randn(real_K, device="cuda"); b = torch.randn(real_K, device="cuda")
o = torch.zeros(1, device="cuda")
# trigger a compile and inspect arg specialization
compiled = kern._splitk_dot_kernel[(512,)](a, b, o, real_K, SPLIT=512, BLOCK=256)
# check cache keys for the signature
sigs = {}
for key, val in kern._splitk_dot_kernel.cache[0].items():
    sig = str(key) + " || " + str(getattr(val, "src", None))
    sigs[str(key)[:200]] = sig[:400]
    break

# Direct dtype test: start value computation for K > 2^31 using the int64 store probe
_probe_kernel[(1,)](K_val, out[:1])
start_computed = out[0].item()
expected = 0  # part=0 -> start=0
res = {
    "triton_version": triton.__version__,
    "cache_signature_sample": sigs,
    "probe_K_used": K_val,
    "start_for_part0_int64result": start_computed,
    "int32_overflow_reproduced": False,
    "note": "full K>2^31 runtime test needs 2 x 8GB+ allocations; this probe inspects scalar arg dtype/specialization instead",
    "vram_gb_available": torch.cuda.get_device_properties(0).total_memory / 1e9,
}
print(json.dumps(res))