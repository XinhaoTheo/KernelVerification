Q = 3329
def multiply(value, coefficient):
    product = value * coefficient
    correction = (product * 3327) & 65535
    reduced = (product + correction * Q) >> 16
    return reduced - Q if reduced >= Q else reduced
# Exhaustive over all coefficient values that can appear (all residues 0..3328), value 0..3328
bad = None
for coeff in range(Q):
    for value in range(Q):
        r = multiply(value, coeff)
        if r != (value * coeff) % Q:
            bad = (value, coeff, r)
            break
    if bad:
        break
# Also verify the exact set of coefficients actually used by the kernel banks
import sys
sys.path.insert(0, "/root/cases/case_151")
import importlib.util
spec = importlib.util.spec_from_file_location("kk", "/root/cases/case_151/kernel.py")
kk = importlib.util.module_from_spec(spec); spec.loader.exec_module(kk)
used_coeffs = set()
for width in (64, 128):
    used_coeffs.update(kk._make_coefficients(width))
bank_bad = None
for coeff in used_coeffs:
    for value in range(Q):
        if multiply(value, coeff) != (value * coeff) % Q:
            bank_bad = (value, coeff)
            break
    if bank_bad:
        break
print({"exhaustive_bad_pair": bad, "num_bank_coeffs": len(used_coeffs), "bank_bad_pair": bank_bad,
       "max_reduced_before_sub": max(((v*3327) & 65535) for v in range(0,0)) if False else "n/a"})