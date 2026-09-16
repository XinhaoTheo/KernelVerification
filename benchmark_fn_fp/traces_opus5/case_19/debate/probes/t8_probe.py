
import json, importlib.util, inspect, traceback
import torch

res = {}
try:
    spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_19/kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    res["public_names"] = [n for n in dir(m) if not n.startswith("_")]
    res["MAX_FUSED_SIZE"] = getattr(m, "MAX_FUSED_SIZE", None)

    fn = None
    for cand in ["cross_entropy_with_predictions"]:
        if hasattr(m, cand):
            fn = getattr(m, cand); res["entry"] = cand
    if fn is None:
        cands = [n for n in dir(m) if callable(getattr(m, n)) and "cross_entropy" in n and not n.startswith("_")]
        res["cross_entropy_callables"] = cands
        if cands:
            fn = getattr(m, cands[0]); res["entry"] = cands[0]
    res["signature"] = str(inspect.signature(fn)) if fn else None

    import triton
    V = 65536
    BLOCK = min(getattr(m, "MAX_FUSED_SIZE", 32768), triton.next_power_of_2(V))
    res["BLOCK_SIZE_guess"] = BLOCK

    torch.manual_seed(0)
    logits = (torch.randn(3, V, device="cuda", dtype=torch.float32) * 0.1) - 3.0
    lo, hi = 100, 100 + BLOCK
    assert hi < V
    # row 0: exact tie across blocks
    logits[0, lo] = 5.0
    logits[0, hi] = 5.0
    # row 1: unique max in first block (control)
    logits[1, 77] = 7.0
    # row 2: unique max in second block (control)
    logits[2, BLOCK + 555] = 7.0
    target = torch.tensor([3, 4, 5], device="cuda", dtype=torch.long)

    ref_logits = logits.clone()
    ref_pred = torch.argmax(ref_logits, dim=-1)  # torch returns first occurrence
    ref_loss = torch.nn.functional.cross_entropy(ref_logits.float(), target, reduction="none")

    out = fn(logits.clone(), target.clone())
    res["out_type"] = str(type(out))
    if isinstance(out, (tuple, list)):
        res["out_len"] = len(out)
        parts = list(out)
    else:
        parts = [out]
    pred = None
    loss = None
    for p in parts:
        if torch.is_tensor(p):
            if p.dtype in (torch.int32, torch.int64):
                pred = p
            elif loss is None:
                loss = p
    res["pred_raw"] = None if pred is None else pred.detach().cpu().tolist()
    res["ref_pred"] = ref_pred.detach().cpu().tolist()
    res["tie_low_index"] = lo
    res["tie_high_index"] = hi
    if pred is not None:
        pl = pred.detach().cpu().tolist()
        pl = pl if isinstance(pl, list) else [pl]
        res["row0_pred"] = pl[0]
        res["row0_expected_lowest"] = lo
        res["row0_tie_bug"] = (pl[0] == hi)
        res["control_rows_match"] = [pl[1] == ref_pred[1].item(), pl[2] == ref_pred[2].item()]
    if loss is not None:
        l = loss.detach().float().flatten()
        res["loss"] = l.cpu().tolist()[:3]
        res["ref_loss"] = ref_loss.detach().cpu().tolist()
        res["max_abs_loss_err"] = float((l[:3] - ref_loss).abs().max().item())
    # verify tie is exact in fp32
    res["tie_values_equal"] = bool(ref_logits[0, lo].item() == ref_logits[0, hi].item())
    res["row0_max"] = float(ref_logits[0].max().item())
except Exception as e:
    res["error"] = repr(e)
    res["tb"] = traceback.format_exc()[-2000:]

print(json.dumps(res, default=str))
