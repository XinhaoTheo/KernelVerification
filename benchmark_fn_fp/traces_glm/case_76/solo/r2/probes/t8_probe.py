import struct, sys
from decimal import Decimal, getcontext
getcontext().prec = 60
sys.path.insert(0,"/root/cases/case_149")

bad_inputs = [0.0029296875, 0.0068359375, 0.0107421875,
              -0.00146484375, -0.00341796875, -0.00537109375, -0.00927734375]
bad_kernel = [14338, 14340, 14342, 14334, 14332, 14330, 14326]
bad_ref    = [14337, 14339, 14341, 14335, 14333, 14331, 14327]

# high-precision exp via series using Fraction-free Decimal
def exp_dec(d):
    s = Decimal(1); term = Decimal(1); n = 1
    while True:
        term = term * d / n
        s += term
        if abs(term) < Decimal("1e-55"): return s
        n += 1

results = []
for xv, kb, rb in zip(bad_inputs, bad_kernel, bad_ref):
    x = Decimal(xv)
    sig = Decimal(1) / (Decimal(1) + exp_dec(-x))
    # round to nearest fp16 manually
    packed = struct.unpack("<e", struct.pack("<e", float(sig)))[0] if False else None
    # find neighbors: decode the two candidate fp16 codes
    def decode(code): return struct.unpack("<e", struct.pack("<H", code & 0xFFFF))[0]
    lo, hi = min(decode(kb), decode(rb)), max(decode(kb), decode(rb))
    midpoint = (Decimal(lo) + Decimal(hi)) / 2
    correct = lo if sig < midpoint else (hi if sig > midpoint else "TIE")
    correct_code = struct.unpack("<H", struct.pack("<e", lo if sig <= midpoint else hi))[0] if correct != "TIE" else None
    results.append({"x": xv, "exact_sig_str": str(sig)[:24], "midpoint": str(midpoint)[:24],
                    "kernel_code": kb, "ref_code": rb, "correct_code_lo": struct.unpack('<H', struct.pack('<e', lo))[0],
                    "exact_below_midpoint": bool(sig < midpoint)})
print(results)
