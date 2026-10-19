
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_30/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}

# Case A: explicit ties, fp16 (reduced precision as contract describes)
logits = torch.tensor([
    [1.0, 3.0, 3.0, 2.0],   # tie at 1,2 -> expect 1
    [5.0, 5.0, 5.0, 5.0],   # all tie -> expect 0
    [0.0, 1.0, 2.0, 3.0],   # unique -> expect 3
    [7.0, 1.0, 7.0, 0.0],   # tie at 0,2 -> expect 0
], device="cuda", dtype=torch.float16)
out = m.route_top1(logits).cpu().tolist()
ref = torch.argmax(logits.float(), dim=1).cpu().tolist()  # torch argmax = lowest index on ties
res["caseA_kernel"] = out
res["caseA_expected_lowest"] = ref

# Case B: random fp16 logits quantized coarsely so ties are common
x = (torch.randn(4096, 8, device="cuda")*2).round().to(torch.float16)
o = m.route_top1(x).cpu()
# expected: lowest index of max
mx = x.float().max(dim=1, keepdim=True).values
expected = (x.float() == mx).float().argmax(dim=1).cpu()
mism = (o.long() != expected).sum().item()
tied = ((x.float()==mx).sum(dim=1) > 1).sum().item()
res["caseB_num_rows"] = x.shape[0]
res["caseB_rows_with_ties"] = tied
res["caseB_mismatches_vs_lowest"] = mism
# also check if kernel matches HIGHEST-index rule
idxs = torch.arange(8, device="cuda").expand_as(x)
highest = torch.where(x.float()==mx, idxs, torch.full_like(idxs,-1)).max(dim=1).values.cpu()
res["caseB_mismatches_vs_highest"] = (o.long()!=highest).sum().item()
# value correctness (selected value equals max)
sel = x.float().gather(1, o.long().cuda().unsqueeze(1)).squeeze(1)
res["caseB_selected_value_wrong"] = (sel != mx.squeeze(1)).sum().item()
res["metric"] = "exact index under lowest-index tie rule (contract-specified)"
print(json.dumps(res, indent=2))
