import torch, struct, json, math
import importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/cases/case_149/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)

# all finite FP16 bit patterns
vals = []
for b in range(0x10000):
    x = struct.unpack('<e', b.to_bytes(2, 'little'))[0]
    if math.isfinite(x):
        vals.append(b)
xs = torch.frombuffer(bytearray(struct.pack('<%dH' % len(vals), *vals)), dtype=torch.float16).clone().cuda()
# torch.frombuffer of uint16 view as fp16: build directly
xs = torch.tensor([struct.unpack('<e', b.to_bytes(2,'little'))[0] for b in vals], dtype=torch.float16).cuda()
assert xs.numel() == len(vals)
snap = xs.clone()
out = km.run(xs)
assert torch.equal(xs, snap), "input mutated"
ref = km.reference(xs.cpu()).cuda()
bits_out = out.view(torch.int16).to(torch.int32).cpu().numpy()
bits_ref = ref.view(torch.int16).to(torch.int32).cpu().numpy()
mism = (bits_out != bits_ref)
mism_idx = [i for i in range(len(vals)) if mism[i]]
examples = [{"input_fp16": float(xs.cpu()[i]), "kernel_bits": int(bits_out[i]) & 0xFFFF,
             "ref_bits": int(bits_ref[i]) & 0xFFFF, "kernel_val": float(out.cpu()[i]),
             "ref_val": float(ref.cpu()[i])} for i in mism_idx[:20]]
# ulp distance
def ulp16(v):
    return struct.unpack('<e', struct.pack('<e', v))[0]
print(json.dumps({"num_inputs_tested": len(vals), "num_bit_mismatches": len(mism_idx),
                 "mismatch_examples": examples, "metric": "exact fp16 bit pattern vs fp64->RNE binary16 reference (exhaustive finite grid)"}))