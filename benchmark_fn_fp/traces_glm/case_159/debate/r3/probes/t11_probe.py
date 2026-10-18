
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
info = {"torch": torch.__version__}
g = torch.Generator("cpu").manual_seed(3)
m = torch.randint(-1, 2, (4, 16, 2, 2), generator=g, dtype=torch.int8).to(dev)
b = torch.randint(-32768, 32768, (4, 16, 2), generator=g, dtype=torch.int16).to(dev)
i = torch.randint(-32768, 32768, (4, 2), generator=g, dtype=torch.int16).to(dev)
# directly probe is_neg/is_conj on int dtypes
try:
    info["is_neg_ok"] = not m.is_neg(); info["is_conj_ok"] = not m.is_conj()
except Exception as ex:
    info["flag_error"] = f"{type(ex).__name__}: {ex}"
try:
    out = k.run(m, b, i)
    ref = k.reference(m, b, i)
    info["run_ok"] = True
    info["exact"] = bool(torch.equal(out, ref))
except Exception as ex:
    info["run_error"] = f"{type(ex).__name__}: {ex}"
    info["run_ok"] = False
print(json.dumps({"claim": "c3", **info}))
