
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_143/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = {}
for N in (32, 64, 128):
    g = torch.Generator(device="cpu").manual_seed(777 + N)
    x = torch.randint(-32768, 32768, (4, N), generator=g, dtype=torch.int32).to("cuda", torch.int16)
    out = k.run(x); ref = k.reference(x)
    # check negated / row-negated variants
    full_neg = torch.equal(out, -ref)                      # out == -ref entirely
    neg_rows = int(((out == -ref) & (ref != 0)).all(dim=1).sum())  # rows equal to negated ref
    exact = torch.equal(out, ref)
    # per-row sign agreement on nonzero coefficients
    nz = ref != 0
    sign_agree = bool((torch.sign(out[nz]) == torch.sign(ref[nz])).all())
    res[f"N{N}"] = {"exact": exact, "full_negation": full_neg, "fully_negated_rows": neg_rows, "sign_agreement_all": sign_agree}
print(json.dumps(res))
