import sys, torch, json
sys.path.insert(0, "/root/cases/case_140")
import kernel as k
dev = "cuda"
# edge residue v ≡ 6 mod 7 near extremes, exact multiples, 0, max, and values around every k*7-1 boundary at top of range
vals = [0,1,6,7,13,4294967289,4294967290,4294967291,4294967292,4294967293,4294967294,4294967295,
        4294967295-6, (4294967296//7-1)*7+6 if False else 4294967290]
# all 7k-1 values at the top: v in [4294967288..4294967295] covers residues 0..6 of largest multiples
vals += [ ( (1<<32)//7 - 1 )*7 + r for r in range(7) ]  # ~613566754*7 boundary region
x = torch.tensor(vals, dtype=torch.int64).to(torch.uint32).to(dev)
out = k.run(x).to(torch.int64)
v = x.to(torch.int64)
row, col = out[:,0], out[:,1]
bad = []
for i in range(x.numel()):
    if not (7*row[i]+col[i] == v[i] and col[i] >= 0 and col[i] <= 6):
        bad.append(int(v[i]))
print(json.dumps({"claim":"c2","n": x.numel(), "violations": bad,
 "row_max_val": int(row[-1]), "col_max_val": int(col[-1]),
 "max_val_row_expected": (4294967295)//7, "max_val_row_got": int(row[-1])}))