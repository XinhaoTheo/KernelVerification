import zlib, sys
sys.path.insert(0, "/root/cases/case_152")
import kernel as K

POLY=0xEDB88320
def advance_bits(state, nbits):
    for _ in range(nbits):
        state = (state >> 1) ^ (POLY if state & 1 else 0)
    return state

def expected_coeffs(width):
    # factor for chunk c: remainder of width-4(c+1) bytes of zeros advanced through init? 
    # The composition convention: contribution_c = polyprod(factor_c, state_c).
    # Ground truth via linearity: CRC(data) with init I = XOR over chunks of A^(8*rem)(fold(chunk)) ^ A^(width*8)(I)... 
    # Simplest ground-truth check: emulate the chunk scheme with a direct iterated advance.
    # factor_c should act on state_c like advancing state_c by 8*(width-4(c+1)) bits, in the polyprod sense.
    # polyprod(L,R) with identity 0x80000000: define D(state) = state advanced one bit, then polyprod(0x80000000,s)=s means identity.
    # Check: advance_factor(k) == polyprod representation of D^(8k). Verify by brute force:
    # for a set of test states s, polyprod(advance_factor(k), s) must equal advance_bits(s, 8*k).
    return None

ok = True
mismatches = []
for width in (64,128):
    factors = K._make_coefficients(width)
    for c in range(width//4):
        k = width - 4*(c+1)
        f = factors[c]
        # identity check: polyprod(0x80000000, s) == s
        # verify factor acts as advancing by 8k bits
        for s in (0, 1, 0xFF, 0xDEADBEEF, 0x80000000, 0xFFFFFFFF, 0x12345678):
            lhs = K._polynomial_product(f, s)
            rhs = advance_bits(s, 8*k)
            if (lhs & 0xFFFFFFFF) != rhs:
                ok = False
                mismatches.append((width, c, k, s, hex(lhs), hex(rhs)))
    # identity
    for s in (1, 0xDEADBEEF, 0xFFFFFFFF):
        if K._polynomial_product(0x80000000, s) & 0xFFFFFFFF != s:
            ok = False
            mismatches.append(("identity", s))
    # correction term check: A^(width*8)(init) ^ MASK
    init = 0xFFFFFFFF
    expected_corr = advance_bits(init, width*8) ^ 0xFFFFFFFF
    if factors[width//4] != expected_corr:
        ok = False
        mismatches.append(("correction", width, hex(factors[width//4]), hex(expected_corr)))

# End-to-end host emulation vs zlib (CPU only, no GPU dependency)
def host_emulate(row):
    width = len(row)
    coeffs = K._make_coefficients(width)
    total = 0
    for c in range(width//4):
        state = 0
        for b in row[4*c:4*c+4]:
            state ^= b
            state = advance_bits(state, 8)
        total ^= K._polynomial_product(coeffs[c], state)
    return total ^ coeffs[width//4]

import random
random.seed(0)
rows = [bytes([0]*64), bytes([0xFF]*64), bytes(random.randrange(256) for _ in range(64)), bytes(128), bytes([0xFF]*128), bytes(random.randrange(256) for _ in range(128))]
host_ok = all((host_emulate(r) == (zlib.crc32(r) & 0xFFFFFFFF)) for r in rows)

import json
print(json.dumps({"factor_checks_pass": ok, "num_mismatches": len(mismatches), "mismatch_examples": mismatches[:3], "host_emulation_matches_zlib": host_ok,
 "metric": "exact 32-bit equality of composition factors vs directly iterated advance; correction term vs iterated init propagation"}))