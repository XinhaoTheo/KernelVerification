import torch, sys, json
sys.path.insert(0, "/root/cases/case_163")
import kernel

dev = "cuda"
lines = []

def run_case(name, v):
    v = v.to(dev)
    ref = kernel.reference(v).double()
    out = kernel.run(v)
    err = (out.double() - ref).abs()
    tol = 1e-4 + 1e-6 * ref.abs()
    bad = (err > tol) | ~torch.isfinite(out)
    lines.append(json.dumps({"case": name, "max_err": err.max().item(),
                             "rel_worst": (err / tol).max().item(),
                             "violations": int(bad.sum()),
                             "nonfinite": int((~torch.isfinite(out)).sum())}))
    return bad

g = torch.Generator(device="cpu").manual_seed(1234)
total_bad = 0
for n in (64, 128, 256):
    for b in (1, 7, 256):
        v = torch.randn((b, n), generator=g).to(torch.float16)
        total_bad += int(run_case(f"rand_n{n}_b{b}", v).sum())

# extreme magnitudes
for n in (64, 128, 256):
    v = torch.full((3, n), 65504.0, dtype=torch.float16)
    v[1] = -65504.0
    v[2] = torch.linspace(-65504, 65504, n).to(torch.float16)
    total_bad += int(run_case(f"extreme_n{n}", v).sum())

# subnormals and signed zeros
v = torch.zeros((4, 128), dtype=torch.float16)
v[0] = 6e-8
v[1] = -6e-8
v[2, 0] = -0.0
v[3] = torch.tensor([x * 6e-8 for x in range(128)]).to(torch.float16)
total_bad += int(run_case("subnormal_signedzero", v).sum())

# adversarial finite bit patterns: sample raw uint16, keep finite only
raw = torch.arange(0, 65536, 97, dtype=torch.int32)
bits = torch.tensor(raw & 0xFFFF, dtype=torch.int16).view(torch.float16)
finite = bits[torch.isfinite(bits) & (torch.abs(bits.view(torch.int16).to(torch.int32) & 0x7F80) != 0x7F80)] if False else bits[torch.isfinite(bits)]
for n in (64, 128, 256):
    f = finite[: 256 * n].reshape(256, n)
    total_bad += int(run_case(f"bitpattern_n{n}", f).sum())

# full-range random magnitudes
v = ((torch.rand((256, 256), generator=g) * 2 - 1) * 65504).to(torch.float16)
total_bad += int(run_case("fullrange_n256_b256", v).sum())

# single-frequency signals (exact integer twiddles) + input preservation check
v = torch.zeros((4, 256), dtype=torch.float16)
for i, freq in enumerate((1, 3, 5, 127)):
    v[i, freq] = 65504.0
    v[i, freq + 1] = -65504.0
orig = v.clone()
total_bad += int(run_case("impulse_n256", v).sum())
pres = bool(torch.equal(orig, v))
lines.append(json.dumps({"input_preserved": pres}))

# output properties
o = kernel.run(v.to(dev))
lines.append(json.dumps({"out_shape": list(o.shape), "out_dtype": str(o.dtype),
                         "out_contig": o.is_contiguous(), "out_device": str(o.device)}))

print("\n".join(lines))
print(json.dumps({"TOTAL_VIOLATIONS": total_bad}))
