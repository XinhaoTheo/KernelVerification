
import json, subprocess, sys, textwrap

child = textwrap.dedent('''
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_21/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
S, MB, PS, HD = 4, 8, 16, 64
NP = 16                       # oversubscribed pool: NP < S*MB = 32
kv = torch.randn(NP, PS, HD, device='cuda', dtype=torch.float32)
# every entry is a legal page id (< NP); sequences share pages (interleaved pool)
bt = torch.randint(0, NP, (S, MB), device='cuda', dtype=torch.int32)
seq_len = MB * PS             # 128, uses all logical blocks
out_info = {"num_pages": NP, "seq_max_physical_touched": S*MB - 1,
            "block_table_max": int(bt.max().item())}
try:
    out = m.paged_gather(kv, bt, seq_len)
    torch.cuda.synchronize()
    t = torch.arange(seq_len, device='cuda')
    pages = bt[:, t // PS].long()
    within = (t % PS).view(1, -1).expand(pages.shape)
    ref = kv[pages, within]
    eq = torch.eq(out, ref)
    out_info.update(launched="ok", cuda_error=None,
                    exact_match_all=bool(eq.all().item()),
                    mismatched_tokens=int((~eq.all(dim=-1)).sum().item()),
                    total_tokens=int(eq.shape[0]*eq.shape[1]),
                    max_abs_err=float((out-ref).abs().max().item()),
                    nonfinite_out=int((~torch.isfinite(out)).sum().item()))
except Exception as e:
    out_info.update(launched="exception", cuda_error=type(e).__name__ + ": " + str(e)[:300])
print("RESULT " + json.dumps(out_info))
''')

p = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=420)
line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
res = json.loads(line[0][7:]) if line else {"launched": "process_died"}
res["child_returncode"] = p.returncode
res["child_stderr_tail"] = p.stderr[-600:]
print(json.dumps(res))
