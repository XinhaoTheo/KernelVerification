
import json, torch, triton, triton.language as tl

@triton.jit
def _route_kernel(Logits, Idx, stride, E: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < E
    x = tl.load(Logits + tok * stride + cols, mask=valid, other=-float("inf"))
    best = -float("inf")
    best_i = 0
    for e in range(E):
        v = tl.sum(tl.where(cols == e, x, 0.0), axis=0)
        take = v >= best
        best = tl.where(take, v, best)
        best_i = tl.where(take, e, best_i)
    tl.store(Idx + tok, best_i)

def route_top1(logits):
    n_tokens, n_experts = logits.shape
    idx = torch.empty(n_tokens, device=logits.device, dtype=torch.int32)
    _route_kernel[(n_tokens,)](logits, idx, logits.stride(0), E=n_experts,
                               BLOCK=triton.next_power_of_2(n_experts))
    return idx

dev='cuda'
rows = [
 [2.0,1.0,2.0],          # tie at 0 and 2
 [3.0,3.0,0.0],          # tie at 0,1
 [1.0,5.0,5.0],          # tie at 1,2
 [7.0,1.0,2.0],          # unique max at 0
 [0.0,1.0,9.0],          # unique max at 2
]
L = torch.tensor(rows, device=dev, dtype=torch.float32)
got = route_top1(L).cpu().tolist()
ref = torch.argmax(L, dim=-1).cpu().tolist()
tied = [i for i,r in enumerate(rows) if r.count(max(r))>1]
untied = [i for i in range(len(rows)) if i not in tied]
res = {
 "metric":"exact_index_equality_vs_torch_argmax",
 "rows":rows, "kernel_idx":got, "ref_idx":ref,
 "tied_row_ids":tied,
 "tied_mismatches":[i for i in tied if got[i]!=ref[i]],
 "untied_mismatches":[i for i in untied if got[i]!=ref[i]],
 "kernel_picks_highest_tied":[ (got[i]==max(j for j,v in enumerate(rows[i]) if v==max(rows[i]))) for i in tied],
 "num_tied":len(tied), "num_tied_mismatch":sum(1 for i in tied if got[i]!=ref[i]),
}
# also non-power-of-2 E check with unique maxima
E=5
torch.manual_seed(0)
L2 = torch.randn(256,E,device=dev,dtype=torch.float32)
g2 = route_top1(L2).cpu(); r2 = torch.argmax(L2,dim=-1).cpu().int()
res["nonpow2_E5_unique_mismatches"]=int((g2!=r2).sum())
print(json.dumps(res))
