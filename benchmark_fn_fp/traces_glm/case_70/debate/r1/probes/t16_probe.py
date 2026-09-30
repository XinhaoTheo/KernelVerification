import sys, json, torch
sys.path.insert(0, "/root/evidence_cases/case_e09")
import kernel

w, v, G = kernel.make_inputs()
G_before = G.clone()
kernel.run(w, v, G)
torch.cuda.synchronize()
G_after = G
byte_equal = bool(torch.equal(G_before.view(torch.uint8), G_after.view(torch.uint8)))
print(json.dumps({"gradients_unchanged_byte_for_byte": byte_equal,
                  "max_abs_diff": float((G_before - G_after).abs().max()),
                  "dtype": str(G.dtype), "shape": list(G.shape)}))