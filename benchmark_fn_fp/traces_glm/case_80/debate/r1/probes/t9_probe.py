import sys, json, random, zlib
sys.path.insert(0, "/root/cases/case_153")
import kernel as K

def composition_crc(row, width):
    # replicate kernel math on host: per 4-byte chunk state from 0, times factor, xor, plus correction
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

res = {}
for width in (64, 128):
    mism = 0; total = 0; first_bad = None
    rng = random.Random(1234 + width)
    cases = [bytes([0])*width, bytes([255])*width, bytes(range(0, 256))[:width]] + [bytes(rng.randrange(256) for _ in range(width)) for _ in range(300)]
    for r in cases:
        total += 1
        if composition_crc(list(r), width) != (zlib.crc32(r) & 0xFFFFFFFF):
            mism += 1
            if first_bad is None: first_bad = r[:16].hex()
    res[f"width{width}"] = {"mismatches": mism, "total": total, "first_bad_prefix": first_bad}
    # also verify correction term / advance factor identity: advancing state 0xFFFFFFFF alone
print(json.dumps({"composition_vs_zlib": res}))
