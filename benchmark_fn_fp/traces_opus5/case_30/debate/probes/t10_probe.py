
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

dev='cuda'; torch.manual_seed(1234)
out={}
for name,dt in [("fp16",torch.float16),("bf16",torch.bfloat16),("fp32",torch.float32)]:
    N,E=8192,8
    L=(torch.randn(N,E,device=dev)*0.5).to(dt)
    got=route_top1(L).cpu().long()
    ref=torch.argmax(L,dim=-1).cpu().long()
    mx=L.max(dim=-1,keepdim=True).values
    tie_count=(L==mx).sum(dim=-1).cpu()
    tied=tie_count>1
    mism=got!=ref
    # verify reduction fidelity: kernel value at chosen idx equals stored logit max
    Lc=L.float().cpu()
    chosen_val=Lc.gather(1,got.unsqueeze(1)).squeeze(1)
    maxval=Lc.max(dim=-1).values
    out[name]={
      "N":N,"E":E,
      "tied_rows":int(tied.sum()),"tied_rate":float(tied.float().mean()),
      "total_mismatch":int(mism.sum()),
      "mismatch_on_tied":int((mism&tied).sum()),
      "mismatch_on_untied":int((mism&~tied).sum()),
      "tied_rows_mismatched_frac":float((mism&tied).float().sum()/max(int(tied.sum()),1)),
      "kernel_chosen_value_always_equals_max":bool(torch.equal(chosen_val,maxval)),
      "max_index_error":int((got-ref).abs().max()),
    }
out["metric"]="exact_index_equality_vs_torch_argmax_on_reduced_precision_logits"
print(json.dumps(out))
