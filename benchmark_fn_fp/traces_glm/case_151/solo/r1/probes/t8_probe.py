import torch, json, sys
sys.path.insert(0, "/root/cases/case_151")

# Exhaustively verify the Shoup-style reduction used in _multiply over all
# (value, coefficient) pairs the kernel can feed it. Values are residues in
# [0,3328]; coefficients in the bank are in [0,3328] (they are R*omega^k mod Q,
# already reduced). Note: the kernel's _multiply returns product*R^{-1} mod Q
# conceptually? Let's just check what it computes: it should be congruent to
# product * R^{-1}... actually check congruence of returned*r vs product? We
# test the semantic the kernel needs: out = (value*coefficient) * R^{-1} mod Q,
# i.e. out ≡ product/65536 (mod 3329), in [0,3328].
R = 1 << 16
Q = 3329
bad = 0
out_of_range = 0
first = None
for value in range(0, 3329):
    for coeff in range(0, 3329):
        product = value * coeff
        correction = (product * 3327) & 65535
        reduced = (product + correction * 3329) >> 16
        if reduced >= 3329:
            out = reduced - 3329
        else:
            out = reduced
        if out < 0 or out > 3328:
            out_of_range += 1
        expect = (product * pow(R, -1, Q)) % Q
        if out != expect:
            bad += 1
            if first is None:
                first = (value, coeff, out, expect)
print(json.dumps({
    "metric": "exact residue equality, all 3329^2 (value,coeff) pairs",
    "pairs_tested": 3329 * 3329,
    "mismatches": bad,
    "out_of_range": out_of_range,
    "first_mismatch": first,
    "note": "kernel applies _multiply to residues in [0,3328] and reduced coefficient bank entries in [0,3328]; reduction result congruent to product*2^-16 mod Q is what the Montgomery-encoded butterflies require (coefficients were premultiplied by R mod Q)."
}))