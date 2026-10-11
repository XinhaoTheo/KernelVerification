import torch, json, sys
sys.path.insert(0, "/root/cases/case_116")
import kernel

dev = "cuda"
# Bias-column identification test: make the matmul product zero (A all zeros,
# legal subnormals/zeros permitted) so out[i,j] == bias[j] exactly; any wrong
# column indexing shows up immediately. Test N=33 with m*n=561 not a multiple
# of 256, plus other N values, across split_k in {1,2,4,8}.
results = []
for (m, n) in [(17,33),(1,1),(2,2),(64,33),(3,64)]:
    for sk in (1,2,4,8):
        k = 33
        a = torch.zeros((m,k), dtype=torch.float16, device=dev)
        b = torch.zeros((k,n), dtype=torch.float16, device=dev)
        bias = torch.linspace(-1, 1, n, dtype=torch.float32, device=dev).clamp(-1,1)
        out = kernel.run(a, b, bias, sk)
        # expected: each column j constant = bias[j]
        exp = bias.unsqueeze(0).expand(m, n)
        err = (out - exp).abs()
        # column-constancy check: does out[i,j] equal bias[j]?
        max_err = float(err.max())
        wrong_cols = int((err.max(dim=0).values > 1e-6).sum())
        results.append({"m":m,"n":n,"k":k,"split_k":sk,"max_bias_err":max_err,
                        "wrong_cols":wrong_cols, "mn_mod_256":(m*n)%256})
print(json.dumps({"metric":"max |out - bias[j]| per element with zeroed matmul",
                  "reason":"c3 claims bias[index % N] misindexes columns for N=33 / m*n not multiple of 256",
                  "cases":results,
                  "any_bias_error":any(r["max_bias_err"]>1e-6 for r in results)}))