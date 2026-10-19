
import json, sys
sys.path.insert(0, "/root/cases/case_151")
import kernel as K

Q = 3329
def multiply_host(value, coefficient):
    product = value * coefficient
    correction = (product * 3327) & 65535
    reduced = (product + correction * 3329) >> 16
    out = reduced - 3329 if reduced >= 3329 else reduced
    return out, reduced, product

# all coefficients the kernel can feed to _multiply: every Montgomery-form residue
coeffs = set()
for width in (64, 128):
    for c in K._make_coefficients(width):
        coeffs.add(c)
# also all possible Montgomery forms of any residue (R*r % Q)
all_mont = [ (1 << 16) * r % Q for r in range(Q) ]
check_coeffs = sorted(coeffs | set(all_mont))

bad = 0
bad_examples = []
max_reduced = 0
max_product = 0
max_after_sub = 0
for v in range(Q):
    for c in check_coeffs:
        out, reduced, product = multiply_host(v, c)
        expected = (v * c) % Q  # since c carries the R factor, correct plain residue is (v*c) mod Q? check montgomery identity
        # Montgomery identity: reduced should equal (v * c * R^-1) mod Q; but kernel uses c = R*w so out = v*w mod Q.
        # So the invariant to test: out == (v * (c * pow(2,16,-1,Q))) % Q
        expected = (v * c * pow(1<<16, -1, Q)) % Q
        max_reduced = max(max_reduced, reduced)
        max_product = max(max_product, product)
        max_after_sub = max(max_after_sub, out)
        if out != expected or not (0 <= out < Q):
            bad += 1
            if len(bad_examples) < 5:
                bad_examples.append([v, c, out, expected, reduced])

# also verify int64/int32 range: max product
print(json.dumps({
  "num_coeffs_checked": len(check_coeffs),
  "bad_count": bad,
  "bad_examples": bad_examples,
  "max_product": max_product,
  "fits_int64": max_product < 2**63,
  "max_reduced_before_sub": max_reduced,
  "reduced_needs_two_subtracts": max_reduced >= 2*Q,
  "max_result": max_after_sub,
  "result_in_range": max_after_sub < Q,
}))
