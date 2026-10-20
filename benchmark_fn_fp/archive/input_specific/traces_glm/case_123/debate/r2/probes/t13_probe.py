import numpy as np, json, sys
sys.path.insert(0, "/root/numerical_cases/case_s")
from kernel import make_inputs_numpy

a, w, p = make_inputs_numpy()
a64, w64, p64 = a.astype(np.float64), w.astype(np.float64), p.astype(np.float64)
t = (np.arange(32, dtype=np.float64) + 0.5) / 32.0

# per-term exact integral and per-term midpoint contribution
exact_terms = a64 * (np.cos(p64) - np.cos(p64 + w64)) / w64          # (4,8)
mid_terms = np.zeros((4, 8))
for r in range(4):
    for k in range(8):
        mid_terms[r, k] = a64[r, k] * np.sin(w64[r, k] * t + p64[r, k]).mean()
err = mid_terms - exact_terms
abs_err = np.abs(err)
dom = np.dstack(np.unravel_index(np.argsort(abs_err.ravel())[::-1][:5], (4, 8)))[0]
rows = []
for r, k in dom:
    rows.append({"row": int(r), "term": int(k), "freq": float(w64[r, k]),
                 "per_term_error": float(err[r, k]),
                 "exact_term": float(exact_terms[r, k]),
                 "ratio_abs_err_to_exact": float(abs(err[r, k]) / max(abs(exact_terms[r, k]), 1e-15))})
total_l2 = np.linalg.norm(err)  # == total quadrature error L2 (rows independent)
print(json.dumps({
    "metric": "per-term midpoint error decomposition on seeded inputs",
    "top5_error_terms": rows,
    "total_quadrature_l2": float(total_l2),
    "n_terms_freq_above_50": int((w64 > 50).sum()),
    "n_terms_freq_above_16": int((w64 > 16).sum()),
    "max_abs_per_term_error": float(abs_err.max()),
    "max_ratio_err_to_own_exact_term": max(x["ratio_abs_err_to_exact"] for x in rows),
}))
