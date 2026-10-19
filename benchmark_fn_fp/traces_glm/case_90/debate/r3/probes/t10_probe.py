import sys, json
sys.path.insert(0, "/root/cases/case_163")
import kernel, torch

dev = "cuda"
results = {}
for N in (64, 128, 256):
    torch.manual_seed(7 + N)
    vals = torch.randn(6, N, device=dev).to(torch.float16).contiguous()
    out = kernel.run(vals)
    ref = torch.fft.rfft(vals.to(torch.float64), dim=1, norm="backward")
    ref = torch.stack((ref.real, ref.imag), dim=-1)
    act_last = out[:, -1, :].to(torch.float64)
    ref_last = ref[:, -1, :]
    tol = 1e-4 + 1e-6 * ref_last.abs()
    # also direct expected values: sum even - sum odd, imag 0
    exp_re = (vals.to(torch.float64)[:, 0::2].sum(1) - vals.to(torch.float64)[:, 1::2].sum(1))
    results[f"N{N}"] = {
        "nyq_re_viol_ratio": float(((act_last[:,0]-ref_last[:,0]).abs()/tol[:,0]).max()),
        "nyq_im_viol_ratio": float(((act_last[:,1]-ref_last[:,1]).abs()/tol[:,1]).max()),
        "nyq_im_max_abs": float(act_last[:,1].abs().max()),
        "nyq_re_vs_evenodd_viol": float((act_last[:,0]-exp_re).abs().max()),
        "ref_last": [[float(a), float(b)] for a, b in ref_last.tolist()],
        "act_last": [[float(a), float(b)] for a, b in act_last.tolist()],
    }
print(json.dumps(results))
