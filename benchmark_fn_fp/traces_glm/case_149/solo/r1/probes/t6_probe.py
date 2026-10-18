import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel

# all finite fp16 bit patterns
bits = np.arange(65536, dtype=np.uint16)
allf = np.frombuffer(bits.tobytes(), dtype="<f2")
finite_mask = np.isfinite(allf)
vals = allf[finite_mask]
print("num finite fp16:", vals.size)

x = torch.from_numpy(vals.copy()).cuda()
out = kernel.run(x).cpu().numpy()

# correct single-rounding reference: fp64 sigmoid -> fp16 RNE
ref = np.empty(vals.shape, dtype="<f2")
v64 = vals.astype(np.float64)
# sigmoid in fp64
sig = 1.0/(1.0+np.exp(-v64))
ref = sig.astype("<f2")

mism = np.count_nonzero(out.view(np.uint16) != ref.view(np.uint16))
idx = np.nonzero(out.view(np.uint16) != ref.view(np.uint16))[0]
extra = {}
if mism:
    i = idx[:10]
    extra = {"bad_inputs": vals[i].tolist(), "kernel_out": out[i].tolist(),
             "ref_out": ref[i].tolist(),
             "absdiff": (out[i].astype(np.float64)-ref[i].astype(np.float64)).tolist(),
             "fp64_value": sig[i].tolist()}
print(json.dumps({"metric":"exact fp16 bit pattern vs fp64-single-rounded reference over ALL finite fp16 inputs",
 "n_tested": int(vals.size), "mismatches": int(mism), **extra}))