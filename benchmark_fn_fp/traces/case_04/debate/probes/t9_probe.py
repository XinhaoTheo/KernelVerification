
import json, sys, torch
sys.path.insert(0, '/root/cases/case_04')
from kernel import rms_norm_forward

torch.manual_seed(0)
res = {}
for name, dt in [('fp32', torch.float32), ('bf16', torch.bfloat16), ('fp16', torch.float16)]:
    X = torch.randn(64, 512, device='cuda').to(dt)
    Y = rms_norm_forward(X, 1e-6)
    x32 = X.float()
    ref = (x32 * torch.rsqrt((x32*x32).mean(-1, keepdim=True) + 1e-6)).to(dt)
    err = None
    try:
        torch.testing.assert_close(Y, ref)
        raised = False
    except Exception as e:
        raised = True
        err = type(e).__name__ + ': ' + str(e).splitlines()[0]
    res[name] = {
        'in_dtype': str(X.dtype), 'out_dtype': str(Y.dtype),
        'dtype_matches_input': (Y.dtype == X.dtype),
        'assert_close_vs_dtype_matched_ref_raised': raised,
        'assert_close_msg': err,
    }
print(json.dumps(res))
