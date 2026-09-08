
import json, importlib.util, traceback
import torch, triton

res = {"cases": []}
try:
    spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_19/kernel.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    fn = m.cross_entropy_with_predictions
    MFS = m.MAX_FUSED_SIZE

    def run(V, tie_idx, label):
        BLOCK = min(MFS, triton.next_power_of_2(V))
        torch.manual_seed(1)
        x = (torch.randn(1, V, device="cuda", dtype=torch.float32) * 0.05) - 4.0
        for j in tie_idx:
            x[0, j] = 3.5
        tgt = torch.tensor([0], device="cuda", dtype=torch.long)
        ref_pred = int(torch.argmax(x[0]).item())
        ref_loss = float(torch.nn.functional.cross_entropy(x.float(), tgt, reduction="none")[0].item())
        loss, pred = fn(x.clone(), tgt.clone())
        p = int(pred.flatten()[0].item())
        vals = [float(x[0, j].item()) for j in tie_idx]
        return {
            "label": label, "V": V, "BLOCK_SIZE": BLOCK,
            "tie_indices": list(tie_idx),
            "tie_blocks": [j // BLOCK for j in tie_idx],
            "tie_values_all_equal": len(set(vals)) == 1,
            "kernel_pred": p, "expected_lowest": min(tie_idx), "torch_argmax": ref_pred,
            "correct": p == min(tie_idx),
            "kernel_loss": float(loss.flatten()[0].item()), "ref_loss": ref_loss,
        }

    B = MFS  # 32768
    # 1. within-block tie only (single block row): should be correct
    res["cases"].append(run(1024, [10, 900], "within_block_single_block_row"))
    # 2. multi-block row, tie entirely inside block 0
    res["cases"].append(run(2*B, [5, 7000], "multiblock_tie_inside_block0"))
    # 3. multi-block row, tie entirely inside block 1
    res["cases"].append(run(2*B, [B+11, B+9000], "multiblock_tie_inside_block1"))
    # 4. cross-block tie, low index near start of block0
    res["cases"].append(run(2*B, [3, B+3], "crossblock_tie_low_early"))
    # 5. cross-block tie, low index late in block0, high index early in block1
    res["cases"].append(run(2*B, [B-1, B], "crossblock_tie_adjacent"))
    # 6. three-way tie spanning 3 blocks
    res["cases"].append(run(3*B, [42, B+42, 2*B+42], "crossblock_tie_three_blocks"))
    # 7. non-power-of-2 vocab, cross-block tie
    res["cases"].append(run(B + 5000, [17, B+123], "crossblock_tie_ragged_last_block"))

    res["n_cases"] = len(res["cases"])
    res["n_incorrect"] = sum(1 for c in res["cases"] if not c["correct"])
    res["incorrect_labels"] = [c["label"] for c in res["cases"] if not c["correct"]]
    res["loss_ratio_ref_over_kernel"] = [round(c["ref_loss"] / c["kernel_loss"], 6) for c in res["cases"]]
except Exception as e:
    res["error"] = repr(e); res["tb"] = traceback.format_exc()[-2000:]

print(json.dumps(res, default=str))
