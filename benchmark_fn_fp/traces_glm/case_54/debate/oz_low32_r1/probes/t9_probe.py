
import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_s/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, f, p = k.make_inputs("cuda")
out = k.run(a, f, p)
A = a.double().cpu().numpy(); F = f.double().cpu().numpy(); P = p.double().cpu().numpy()
ref = 1 + (A*(np.cos(P)-np.cos(P+F))/F).sum(axis=1)
o = out.double().cpu().numpy()
row_err = np.abs(o-ref)
# per-term contribution breakdown: term r,k sampled avg vs exact integral contribution
pts = (np.arange(32)+0.5)/32.0
sine_avg = np.sin(np.multiply.outer(np.arange(4)*0, pts)+pts*0)  # placeholder
term_err = np.zeros((4,8))
for r in range(4):
    for kk in range(8):
        avg = np.mean(np.sin(F[r,kk]*pts+P[r,kk]))
        exact = (np.cos(P[r,kk])-np.cos(P[r,kk]+F[r,kk]))/F[r,kk]
        term_err[r,kk] = abs(A[r,kk]*(avg-exact))
print(json.dumps({"row_abs_err": row_err.tolist(), "row_err_over_0p05": bool((row_err>0.05).any()),
                  "freqs": F.tolist(), "max_freq": float(F.max()),
                  "rows_with_freq_gt100": [bool((F[r]>100).any()) for r in range(4)],
                  "term_abs_err_max": float(term_err.max()),
                  "max_term_err_freq": float(F.flatten()[int(term_err.argmax())])}))
