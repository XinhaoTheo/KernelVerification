"""Validate exposed inputs and CPU witnesses without importing Triton or torch."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .build import measure
except ImportError:
    from build import measure

ROOT = Path(__file__).resolve().parents[2] / "correlation_pair"


class Array:
    def __init__(self, x): self.x = x
    def to(self, device): return self.x


class Adapter:
    from_numpy = Array


def sequential(x, w):
    scale = np.max(np.abs(w), axis=1, keepdims=True) / np.float32(7)
    q = np.clip(np.floor(w/scale + np.float32(0.5)), -7, 7)
    products = ((q*scale).astype(np.float32)*x).astype(np.float32)
    result = np.zeros(w.shape[0], dtype=np.float32)
    for j in range(w.shape[1]):
        result = (result + products[:, j]).astype(np.float32)
    return result


def main():
    private = ROOT / "private_data"
    key = json.loads((private/"answer_key_cpu.json").read_text())
    outputs = {}
    data = {}
    for name, expected in key["cases"].items():
        path=ROOT.parent/"triton_eval_cases"/name/"kernel.py"
        code=path.read_text()
        assert hashlib.sha256(code.encode()).hexdigest()==expected["kernel_sha256"]
        tree=ast.parse(code)
        # Execute the actual verifier-visible generator and constants, not a
        # separately maintained approximation of them. Adapter only removes .to().
        nodes=[n for n in tree.body if isinstance(n,ast.Assign) or
               (isinstance(n,ast.FunctionDef) and n.name=="make_inputs")]
        ns={"torch":Adapter}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns)
        x,a,b=ns["make_inputs"]("cpu")
        for array,label,sha in zip((x,a,b),("x","a","b"),expected["input_sha256"]):
            assert np.isfinite(array).all()
            assert array.dtype==np.float32
            assert hashlib.sha256(array.tobytes()).hexdigest()==sha
            with np.load(private/f"{name}_inputs.npz") as artifact:
                assert np.array_equal(array,artifact[label])
        actual=measure(x,a,b,np.arange(len(a)))
        assert actual["cpu_ground_truth"]==expected["cpu_ground_truth"]
        assert abs(actual["error"]-expected["error"])<1e-12
        reference=np.sum(a.astype(np.float64)*x.astype(np.float64),axis=1)
        reference+=np.sum(b.astype(np.float64)*x.astype(np.float64),axis=1)
        seq=(sequential(x,a)+sequential(x,b)).astype(np.float32)
        seq_error=float(np.linalg.norm(seq.astype(np.float64)-reference)/np.linalg.norm(reference))
        assert (seq_error<=key["budget"])==(actual["error"]<=key["budget"])
        assert abs(seq_error-actual["error"])<=actual["reduction_order_error_bound"]
        outputs[name]={"cpu_vectorized_error":actual["error"],"cpu_sequential_error":seq_error,
                       "label":actual["cpu_ground_truth"],"exposed_inputs_match":True}
        data[name]=(x,a,b)
    left,right=data.values()
    assert np.array_equal(left[0],right[0]) and np.array_equal(left[1],right[1])
    # Exact row multisets, not just approximately matching mean and variance.
    assert sorted(row.tobytes() for row in left[2])==sorted(row.tobytes() for row in right[2])
    result={"checks_passed":True,"gpu_verified":False,"model_evaluated":False,"cases":outputs}
    (private/"validation_cpu.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
