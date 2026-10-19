import torch, sys, json, random
sys.path.insert(0, "/root/cases/case_112")
import kernel as K

random.seed(7)
dev = "cuda"
configs = []
for B in (1, 2):
    for L in (1, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 128, 129, 255, 257):
        for H in (1, 3, 4):
            for D in (1, 8, 96):
                for Ksz in (16, 32, 64):
                    configs.append((B, L, H, D, Ksz))
random.shuffle(configs)
test_set = []
# deterministic coverage + random extras
for B in (1,2):
    for L in (1,16,17,32,33,64,65,97,257):
        for H in (1,2,4):
            for D in (1,33,96):
                for Ksz in (16,32,64):
                    test_set.append((B,L,H,D,Ksz))
test_set += configs[:60]

worst = 0.0; worst_cfg = None; total_viol = 0; ntests = 0
for (B,L,H,D,Ksz) in test_set:
    g = torch.Generator(device="cpu").manual_seed(ntests*13+5)
    u = (2*torch.rand((B,L,H,D), generator=g)-1).float().to(dev)
    dec = (0.5+0.46875*torch.rand((B,L,H), generator=g)).float().to(dev)
    init = (2*torch.rand((B,H,D), generator=g)-1).float().to(dev)
    seq = torch.zeros((B,L), dtype=torch.int32)
    # random nondecreasing labels
    for b in range(B):
        nres = random.randint(0, 5)
        for _ in range(nres):
            pos = random.randint(1, max(1,L-1))
            seq[b, pos:] += random.randint(1,3)
    seq = torch.clamp(seq, max=L-1).to(dev)
    out, final = K.run(u, dec, seq, init, Ksz)
    ro, rf = K.reference(u, dec, seq, init, Ksz)
    ro = ro.float(); rf = rf.float()
    viol = ((out-ro).abs() > 0.002+0.0001*ro.abs()).sum().item()
    viol += ((final-rf).abs() > 0.002+0.0001*rf.abs()).sum().item()
    err = (out-ro).abs().max().item()
    total_viol += viol; ntests += 1
    if err > worst:
        worst = err; worst_cfg = (B,L,H,D,Ksz)
print(json.dumps({
  "metric": "elementwise |actual-target| vs contract tolerance",
  "num_tests": ntests,
  "total_violations": total_viol,
  "max_abs_err": worst,
  "worst_cfg": worst_cfg,
}))