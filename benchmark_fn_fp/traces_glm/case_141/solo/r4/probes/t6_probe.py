import torch, sys
sys.path.insert(0, "/root/cases/case_141")
import kernel as K

CH = 131072
total = 1 << 32
mismatches = []
checked = 0
bad_examples = []
for base in range(0, total, CH):
    n = min(CH, total - base)
    idx = torch.arange(base, base + n, dtype=torch.int64).to(torch.uint32).cuda()
    out = K.run(idx)
    vals = idx.to(torch.int64)
    ref = torch.stack((vals // 7, vals % 7), dim=1).to(torch.uint32)
    if not torch.equal(out, ref):
        bad = (out != ref).any(dim=1)
        nb = int(bad.sum())
        mismatches.append((base, nb))
        if len(bad_examples) < 5:
            bidx = bad.nonzero()[:3].flatten()
            for j in bidx.tolist():
                bad_examples.append((int(idx[j]), int(out[j,0]), int(out[j,1]), int(ref[j,0]), int(ref[j,1])))
    checked += n

# also targeted extremes: multiples of 7 and neighbors near max
special = torch.tensor([0,1,6,7,8,2**31-1,2**31,2**31+6,2**32-8,2**32-7,2**32-6,2**32-1,
                        7*(2**32//7), 7*(2**32//7)-1, 7*(2**32//7)+1], dtype=torch.int64).to(torch.uint32).cuda()
outs = K.run(special)
refs = torch.stack((special.to(torch.int64)//7, special.to(torch.int64)%7), dim=1).to(torch.uint32)
special_ok = torch.equal(outs, refs)

print({"checked": checked, "mismatch_chunks": mismatches[:10], "num_mismatch_chunks": len(mismatches),
       "bad_examples": bad_examples, "special_ok": special_ok,
       "special": special.cpu().tolist(), "special_out": outs.cpu().tolist(), "special_ref": refs.cpu().tolist()})
import json
print(json.dumps({"metric":"exact elementwise equality vs int64 division over full uint32 domain","checked":checked,
 "num_mismatch_chunks":len(mismatches),"first_mismatches":mismatches[:10],"bad_examples":bad_examples,"special_ok":special_ok}))
