
import json, importlib.util, traceback, torch
spec=importlib.util.spec_from_file_location("k","/root/cases/case_22/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows=[]; worst_count_dev=0.0; worst_rel=0.0
Ks=[512, 513, 1023, 1537, 100003, 131072, 131073, 131071, 512*256, 512*256+1, 999983, 1000003, 1048577]
torch.manual_seed(1)
for K in Ks:
    o=torch.ones(K,device="cuda",dtype=torch.float32)
    cnt=m.splitk_dot(o,o).item()   # equals number of counted terms exactly for K<2^24
    a=torch.randn(K,device="cuda",dtype=torch.float32)
    b=torch.randn(K,device="cuda",dtype=torch.float32)
    v=m.splitk_dot(a,b).item()
    ref=torch.dot(a.double(),b.double()).item()
    sumabs=torch.sum((a.double()*b.double()).abs()).item()
    rel=abs(v-ref)/sumabs
    chunk=-(-K//512)
    rows.append({"K":K,"chunk":chunk,"K_mod_512":K%512,"ones_count":cnt,"count_minus_K":cnt-K,
                 "randn_val":v,"fp64_ref":ref,"rel_err_vs_sumabs":rel})
    worst_count_dev=max(worst_count_dev,abs(cnt-K)); worst_rel=max(worst_rel,rel)
print(json.dumps({"rows":rows,"worst_abs_count_deviation":worst_count_dev,
                  "worst_rel_err_vs_sumabs":worst_rel,
                  "metric":"ones-vector result == K proves exact term coverage; rel err normalized by sum|a*b| is cancellation-aware fp32 noise measure"}))
