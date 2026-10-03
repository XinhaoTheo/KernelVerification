import torch, sys, json
sys.path.insert(0, "/root/cases/case_112")
import kernel as K

dev = "cuda"
torch.manual_seed(1)
B, L, H, D = 1, 64, 2, 8
chunk = 32
u = (2*torch.rand(B,L,H,D)-1).float().to(dev)
dec = (0.5 + 0.46*torch.rand(B,L,H)).float().to(dev)
init = (2*torch.rand(B,H,D)-1).float().to(dev)
seq = torch.zeros(B,L, dtype=torch.int32)
# mid-chunk reset at token 10 (inside chunk 0), another at token 40 (chunk boundary of chunk 1)
seq[:, 10:] += 1
seq[:, 40:] += 1
seq = seq.to(dev)

out, final = K.run(u, dec, seq, init, chunk)
ref_out, ref_final = K.reference(u, dec, seq, init, chunk)
ref_out = ref_out.float(); ref_final = ref_final.float()

diff = (out - ref_out).abs()
tol = 0.002 + 0.0001*ref_out.abs()
viol = diff > tol
# examine tokens 10..31 (after first reset, still chunk 0)
bad = viol[0,:,0,:]
tok_bad = bad.any(dim=1).nonzero().flatten().tolist()
print(json.dumps({
  "metric": "elementwise |actual-target| vs contract tolerance",
  "max_abs_err": diff.max().item(),
  "num_violations": int(viol.sum().item()),
  "tokens_with_violations": tok_bad[:40],
  "max_err_token10_31": diff[0,10:32].max().item(),
  "max_err_token32_39": diff[0,32:40].max().item(),
  "final_state_max_err": (final-ref_final).abs().max().item(),
  "seq_boundaries": [10,40],
}))
