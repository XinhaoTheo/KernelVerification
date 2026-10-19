import sys, json, torch
sys.path.insert(0, "/root/cases/case_153")
import kernel as K

def host_composition_crc(row, width):
    coeffs = K._make_coefficients(width)
    total = 0
    for chunk in range(width // 4):
        state = 0
        for byte in row[chunk*4:(chunk+1)*4]:
            state ^= byte
            for _ in range(8):
                state = (state >> 1) ^ (K._POLY if state & 1 else 0)
        total ^= K._polynomial_product(coeffs[chunk], state)
    return total ^ coeffs[width // 4]

out = {"total_rows": 0, "parity_mismatches": 0, "per_width": {}}
for width in (64, 128):
    mism = 0; rows = 0
    for seed in range(4):
        gen = torch.Generator(device="cpu").manual_seed(seed * 31 + 3)
        x = torch.randint(0, 256, (128, width), generator=gen, dtype=torch.uint8).cuda()
        got = K.run(x).cpu().tolist()
        for i, r in enumerate(x.cpu().tolist()):
            rows += 1
            if got[i] != host_composition_crc(r, width):
                mism += 1
    out["per_width"][f"width{width}"] = {"rows": rows, "parity_mismatches": mism}
    out["total_rows"] += rows; out["parity_mismatches"] += mism
print(json.dumps(out))
