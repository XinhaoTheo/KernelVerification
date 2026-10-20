import json, sys, torch
sys.path.insert(0, '/root/cases/case_167')
import kernel

x = torch.arange(256, dtype=torch.uint8, device='cuda')
out = kernel.run(x)
ref = kernel.reference(x)
mism = (out != ref)
bad = [(int(a), int(b), int(c)) for a, b, c in zip(x[mism].tolist(), out[mism].tolist(), ref[mism].tolist())]
print(json.dumps({"mismatch_count": int(mism.sum()), "examples": bad[:10], "s0": int(out[0]), "s_0x53": int(out[0x53]), "out_dtype": str(out.dtype), "out_contig": out.is_contiguous()}))