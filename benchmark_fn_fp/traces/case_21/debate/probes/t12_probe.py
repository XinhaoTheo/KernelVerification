
import json, subprocess, sys, textwrap, importlib.util, torch

res = {}

# ---------- Part A: sentinel backing buffer, identify OOB reads by value ----------
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_21/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
S, MB, PS, HD = 4, 8, 16, 64
NP = 16
buf = torch.randn(64, PS, HD, device='cuda', dtype=torch.float32)   # 64 pages of backing store
kv = buf[:NP]                                                       # kv_cache exposes only 16 pages
assert kv.is_contiguous() and kv.data_ptr() == buf.data_ptr() and kv.stride(0) == PS*HD
bt = torch.randint(0, NP, (S, MB), device='cuda', dtype=torch.int32)  # every entry a legal page id
seq_len = MB * PS   # 128

out = m.paged_gather(kv, bt, seq_len)
torch.cuda.synchronize()

t = torch.arange(seq_len, device='cuda')
lb = t // PS
within = t % PS
phys = (torch.arange(S, device='cuda').view(-1,1) * MB + lb.view(1,-1))   # kernel's synthesized index
oob_mask = phys >= NP                                                     # tokens outside kv_cache
# what the kernel WOULD return if it read buf[phys, within] (i.e. past kv_cache for phys>=16)
pred_oob = buf[phys.reshape(-1).long(), within.view(1,-1).expand(S,-1).reshape(-1)].view(S, seq_len, HD)
eq_pred = torch.eq(out, pred_oob).all(dim=-1)

ref = buf[bt[:, lb].long().reshape(-1), within.view(1,-1).expand(S,-1).reshape(-1)].view(S, seq_len, HD)
eq_ref = torch.eq(out, ref).all(dim=-1)

res["partA"] = {
    "num_pages_exposed": NP,
    "max_synth_physical": int(phys.max().item()),
    "tokens_total": int(phys.numel()),
    "tokens_with_phys_ge_num_pages": int(oob_mask.sum().item()),
    "oob_tokens_matching_backing_buffer_beyond_kv": int((eq_pred & oob_mask).sum().item()),
    "inbounds_tokens_matching_synth_index": int((eq_pred & ~oob_mask).sum().item()),
    "tokens_matching_reference": int(eq_ref.sum().item()),
    "kv_storage_pages": NP,
    "buffer_pages": 64,
}

# ---------- Part B: push far past the allocation, expect a fatal CUDA fault ----------
child = textwrap.dedent('''
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_21/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
S, MB, PS, HD = 64, 4096, 16, 64
NP = 1
kv = torch.randn(NP, PS, HD, device='cuda', dtype=torch.float32)   # 4 KB total
bt = torch.zeros((S, MB), device='cuda', dtype=torch.int32)        # page 0 everywhere: all legal
seq_len = PS                                                       # 16, fits in max_blocks pages
info = {"num_pages": NP, "max_synth_physical": (S-1)*MB,
        "bytes_past_allocation": (S-1)*MB*PS*HD*4}
try:
    out = m.paged_gather(kv, bt, seq_len)
    torch.cuda.synchronize()
    info.update(status="no_fault", exact_match_all=bool(torch.eq(out, kv[0,:seq_len].expand(S,seq_len,HD)).all().item()))
except Exception as e:
    info.update(status="exception", err=type(e).__name__ + ": " + str(e)[:300])
print("RESULT " + json.dumps(info))
''')
p = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=400)
line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
res["partB"] = json.loads(line[0][7:]) if line else {"status": "process_died_before_print"}
res["partB"]["child_returncode"] = p.returncode
res["partB"]["child_stderr_tail"] = p.stderr[-500:]

print(json.dumps(res))
