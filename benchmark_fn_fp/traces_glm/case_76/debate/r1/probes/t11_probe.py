import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/cases/case_149/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)

results = {}
def try_call(name, fn):
    try:
        out = fn()
        results[name] = {"accepted": True, "value": out if not isinstance(out, torch.Tensor) else {"bits": out.view(torch.int16).item()}}
    except Exception as e:
        results[name] = {"accepted": False, "error": type(e).__name__, "msg": str(e)[:120]}

# legal: contiguous view with nonzero storage offset
base = torch.randn(300, dtype=torch.float16).cuda()
view = base[7:263]  # contiguous, offset 7
try_call("legal_offset_view", lambda: (km.run(view), torch.equal(base[7:263], view))[0])
# check output correctness for the view
try:
    o = km.run(view)
    r = km.reference(view)
    results["legal_offset_view_correct"] = {"bit_equal_to_reference": bool((o.view(torch.int16) == r.view(torch.int16)).all().item()), "expected": True}
except Exception as e:
    results["legal_offset_view_correct"] = {"error": str(e)[:120]}

# legal: signed zeros, subnormals, max finite -> covered elsewhere, quick smoke
edge = torch.tensor([0.0, -0.0, 6.104e-5, -6.104e-5, 65504.0, -65504.0], dtype=torch.float16).cuda()
o = km.run(edge); r = km.reference(edge)
results["edge_values_bits_equal"] = {"expected": True, "actual": bool((o.view(torch.int16) == r.view(torch.int16)).all().item()),
                                     "out_bits": o.view(torch.int16).cpu().tolist(), "ref_bits": r.view(torch.int16).cpu().tolist()}

# illegal inputs must be rejected
try_call("nan_rejected", lambda: km.run(torch.tensor([float('nan')], dtype=torch.float16).cuda()))
try_call("inf_rejected", lambda: km.run(torch.tensor([float('inf')], dtype=torch.float16).cuda()))
try_call("noncontig_rejected", lambda: km.run(torch.randn(4, 8, dtype=torch.float16).cuda().t()[0]))
try_call("wrong_dtype_rejected", lambda: km.run(torch.randn(8, dtype=torch.float32).cuda()))
try_call("cpu_rejected", lambda: km.run(torch.randn(8, dtype=torch.float16)))
try_call("empty_rejected", lambda: km.run(torch.zeros(0, dtype=torch.float16).cuda()))
try_call("too_big_rejected", lambda: km.run(torch.zeros(131073, dtype=torch.float16).cuda()))
class SubT(torch.Tensor): pass
try_call("subclass_rejected", lambda: km.run(torch.randn(8, dtype=torch.float16).cuda().as_subclass(SubT)))
try:
    with torch.inference_mode():
        km.run(torch.randn(8, dtype=torch.float16).cuda())
    results["inference_mode_rejected"] = {"accepted": True, "expected": False}
except Exception as e:
    results["inference_mode_rejected"] = {"accepted": False, "error": type(e).__name__}
try:
    g = torch.randn(8, dtype=torch.float16).cuda().requires_grad_(True)
    km.run(g); results["requires_grad_rejected"] = {"accepted": True, "expected": False}
except Exception as e:
    results["requires_grad_rejected"] = {"accepted": False, "error": type(e).__name__}

print(json.dumps(results))