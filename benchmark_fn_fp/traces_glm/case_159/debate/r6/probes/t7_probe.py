
import torch, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel as K

dev = "cuda"
results = {}
# Crafted: chunk summaries noncommuting. Chunk 0&1 use matrix [[1,0],[0,1]]*2? entries must be -1/0/1.
# Use chunk0 matrix = [[1,1],[0,1]] each step (summary approx [[8?,..]]), chunk2 = [[1,0],[1,1]].
# Simpler: random inputs with several seeds, plus crafted.
def crafted(N, B=2):
    g = torch.Generator().manual_seed(5)
    m = torch.zeros(B, N, 2, 2, dtype=torch.int8)
    m[:] = torch.tensor([[1,1],[0,1]], dtype=torch.int8)  # default A = [[1,1],[0,1]]
    for b in range(B):
        # steps 16..23 (chunk 2) use a noncommuting matrix
        for s in range(16, min(24, N)):
            m[b, s] = torch.tensor([[1,0],[1,1]], dtype=torch.int8)
    bias = torch.randint(-300, 300, (B, N, 2), generator=g, dtype=torch.int16)
    init = torch.randint(-300, 300, (B, 2), generator=g, dtype=torch.int16)
    return m.to(dev), bias.to(dev), init.to(dev)

fails = {}
def check(N, tag):
    m, b, i = crafted(N)
    out = K.run(m, b, i)
    ref = K.reference(m.cpu(), b.cpu(), i.cpu()).to(dev)
    ok = torch.equal(out, ref)
    if not ok:
        mism = (out != ref).any(dim=-1)  # [B,N]
        first = mism.nonzero()
        info = {"first_mismatch_row_step": [int(x) for x in first[0].tolist()] if len(first) else None,
                "n_mismatch_elements": int((out!=ref).sum().item()),
                "out_step24": out[0,24].tolist(), "ref_step24": ref[0,24].tolist(),
                "out_step16": out[0,16].tolist(), "ref_step16": ref[0,16].tolist()}
        fails[tag] = info
    return ok, out, ref

ok32, out32, ref32 = check(32, "N32_crafted")
# also random seeds at N=32 and other N for context
rand_fails = {}
for seed in range(5):
    g = torch.Generator().manual_seed(seed)
    m = torch.randint(-1,2,(4,32,2,2), generator=g, dtype=torch.int8)
    b = torch.randint(-32768,32768,(4,32,2), generator=g, dtype=torch.int16)
    i = torch.randint(-32768,32768,(4,2), generator=g, dtype=torch.int16)
    m,b,i = m.to(dev), b.to(dev), i.to(dev)
    out = K.run(m,b,i); ref = K.reference(m.cpu(),b.cpu(),i.cpu()).to(dev)
    rand_fails[seed] = int((out!=ref).sum().item())

import json
print(json.dumps({
  "metric": "exact elementwise equality (atol=rtol=0) vs sequential int64 reference",
  "crafted_N32_exact_match": ok32,
  "crafted_fail_details": fails,
  "random_N32_mismatch_counts_by_seed": rand_fails,
}))
