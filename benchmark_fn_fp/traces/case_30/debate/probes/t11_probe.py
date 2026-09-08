
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

dev='cuda'; res={}
for E in [2,3,4,5,6,8,16]:
    for dtname,dt in [("fp32",torch.float32),("fp16",torch.float16)]:
        L=torch.full((4,E),0.25,device=dev,dtype=dt)
        got=route_top1(L).cpu().tolist()
        ref=torch.argmax(L,dim=-1).cpu().tolist()
        res[f"E{E}_{dtname}"]={"kernel":got,"ref":ref,"expected_contract":0,
                               "equals_E_minus_1":all(g==E-1 for g in got),
                               "index_error":max(abs(g-r) for g,r in zip(got,ref))}
res["metric"]="all_equal_row_returns_lowest_index_0"
print(json.dumps(res))
