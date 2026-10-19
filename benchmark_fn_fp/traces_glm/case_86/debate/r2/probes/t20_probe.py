import torch, json, importlib.util
import triton, triton.language as tl
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
@triton.jit
def scan_test(INP, OUT6, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    a = tl.load(INP + lane).to(tl.int64)
    z = tl.zeros_like(a)
    o = (0 * a + 1)
    aa, bb, cc, dd, ee, ff = tl.associative_scan((o, z, z, o, a, z), 0, k._compose)
    tl.store(OUT6 + lane * 6, aa)
    tl.store(OUT6 + lane * 6 + 1, bb)
    tl.store(OUT6 + lane * 6 + 2, cc)
    tl.store(OUT6 + lane * 6 + 3, dd)
    tl.store(OUT6 + lane * 6 + 4, ee)
    tl.store(OUT6 + lane * 6 + 5, ff)
@triton.jit
def gather_test(INP, OUT, IDX, C: tl.constexpr):
    lane = tl.arange(0, C)
    idx = tl.load(IDX + lane)
    tl.store(OUT + lane, tl.gather(INP + tl.arange(0, C), idx, 0))
results = {}
inp = torch.arange(1, 9, dtype=torch.int32, device=dev)
out = torch.empty(8 * 6, dtype=torch.int64, device=dev)
scan_test[(1,)](inp, out, BLOCK=8)
expect = torch.cat([torch.ones(8), torch.zeros(8), torch.zeros(8), torch.ones(8),
                    inp.to(torch.int64).cumsum(0).to(dev), torch.zeros(8, device=dev)]).view(8, 6)
results["tuple_scan_ok"] = bool(torch.equal(out.view(8, 6), expect))
for C in [1, 2, 4, 8]:
    inp = torch.arange(C, dtype=torch.int64, device=dev) * 10
    idx = torch.arange(C, dtype=torch.int32, device=dev).flip(0)
    out = torch.empty(C, dtype=torch.int64, device=dev)
    gather_test[(1,)](inp, out, idx, C=C)
    results["gather_ok_C%d" % C] = bool(torch.equal(out, inp[idx]))
err = None
try:
    m, b, i = k.make_inputs(batch=3, length=17)
    results["run_B3_N17_exact"] = bool(torch.equal(k.run(m, b, i), k.reference(m, b, i)))
except Exception as ex:
    err = repr(ex)[:300]
results["error"] = err
print(json.dumps(results))