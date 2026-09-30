import sys, json
sys.path.insert(0, "/root/evidence_cases/case_e08")
import numpy as np, torch
import kernel as K

features, anchors = K.make_inputs()
a = anchors.cpu().numpy()
absa = np.abs(a.astype(np.float64))
ties = []
for v in range(2):
    for i in range(4):
        for j in range(i+1, 4):
            if absa[v, i] == absa[v, j]:
                ties.append([v, i, j, float(absa[v, i])])

# compute slots per view
slots = {}
for v in range(2):
    s = [int((absa[v] < absa[v, c]).sum()) for c in range(4)]
    slots[v] = s

# fill output with NaN sentinel, run kernel, check all slots written
features2, anchors2 = K.make_inputs()
out = K.run(features2, anchors2)
y = out.detach().cpu().numpy().astype(np.float64)
signs = {v: [1 if a[v, c] >= 0 else -1 for c in range(4)] for v in range(2)}
# verify each output slot equals sign*source column (copy coverage check)
covered = True
detail = []
for v in range(2):
    for c in range(4):
        slot = slots[v][c]
        col = y[v, :, slot]
        src = features2.cpu().numpy().astype(np.float64)[v, :, c] * signs[v][c]
        ok = bool(np.allclose(col, src))
        detail.append({"view": v, "src_col": c, "slot": slot, "matches_signed_source": ok})
        covered = covered and ok
# non-written slots would fail to match any source column
result = {
    "metric": "anchor magnitude tie detection within view + per-slot write coverage vs signed source",
    "anchor_values": a.tolist(),
    "anchor_abs": absa.tolist(),
    "ties_in_fixed_draw": ties,
    "per_view_slots": slots,
    "per_view_signs": signs,
    "all_slots_written_and_match": covered,
    "detail": detail,
    "output_finite": bool(np.isfinite(y).all()),
}
print(json.dumps(result))
