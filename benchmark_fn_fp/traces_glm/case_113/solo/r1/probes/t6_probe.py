import torch, json, sys
sys.path.insert(0, "/root/cases/case_113")
import kernel as K

def test(length, chunk, bnd, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B,H,D = 1,2,33
    u = (2*torch.rand((B,length,H,D),generator=g)-1).float().cuda()
    decay = (.90+.06*torch.rand((B,length,H),generator=g)).float().cuda()
    init = (2*torch.rand((B,H,D),generator=g)-1).float().cuda()
    seq = torch.zeros((B,length),dtype=torch.int32)
    seq[:, bnd:] += 1
    seq = seq.cuda()
    out, fin = K.run(u, decay, seq, init, chunk)
    ref_out, ref_fin = K.reference(u, decay, seq, init, chunk)
    ref_out = ref_out.float().cuda(); ref_fin = ref_fin.float().cuda()
    tol_out = 0.002 + 0.0001*ref_out.abs()
    tol_fin = 0.002 + 0.0001*ref_fin.abs()
    bad_out = (out-ref_out).abs() > tol_out
    bad_fin = (fin-ref_fin).abs() > tol_fin
    i = (out-ref_out).abs().argmax().item()
    return dict(length=length, chunk=chunk, boundary=bnd,
        out_bad=int(bad_out.sum()), out_total=ref_out.numel(),
        fin_bad=int(bad_fin.sum()),
        max_abs_err=float((out-ref_out).abs().max()),
        max_fin_err=float((fin-ref_fin).abs().max()),
        worst_idx=i)

results = []
# boundary strictly inside chunk (token 33 with K=32: chunks 0..31,32..63)
for (L,K,b) in [(97,32,33),(97,32,20),(97,64,70),(65,16,40),(97,32,0),(97,32,32)]:
    results.append(test(L,K,b))

print(json.dumps(results, indent=1))
