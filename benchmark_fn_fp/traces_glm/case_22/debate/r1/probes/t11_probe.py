import torch, json, triton
import triton.language as tl

@triton.jit
def _idx_probe(K, SPLIT: tl.constexpr, Start, Idx):
    part = tl.program_id(0)
    chunk = tl.cdiv(K, SPLIT)
    start = part * chunk
    # emulate the kernel's last tile index arithmetic: off = chunk-256 rounded down
    off = ((chunk - 256) // 256) * 256
    idx = start + off + tl.arange(0, 256)
    mask = idx < tl.minimum(start + chunk, K)
    tl.store(Start + part, start.to(tl.int64))
    # store the max index under this mask (0 if none), as int64
    mx = tl.max(tl.where(mask, idx, -2147483648), axis=0)
    tl.store(Idx + part, mx.to(tl.int64))

def check(K):
    starts = torch.zeros(512, device="cuda", dtype=torch.int64)
    idxs = torch.full((512,), -1, device="cuda", dtype=torch.int64)
    _idx_probe[(512,)](K, SPLIT=512, Start=starts, Idx=idxs)
    torch.cuda.synchronize()
    chunk = -(-K // 512)
    exp_start = [p * chunk for p in range(512)]
    got = starts.tolist()
    start_ok = got == exp_start
    # any negative start indicates int32 wraparound
    neg = [v for v in got if v < 0]
    return {"K": K, "chunk": chunk, "start_matches_expected": start_ok,
            "negative_start_count": len(neg),
            "max_start": max(got), "expected_max_start": exp_start[-1],
            "sample_idx_max": int(idxs.max().item())}

res = {
    "triton_version": triton.__version__,
    "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    "case_K_just_over_2p31": check(2**31 + 100),
    "case_K_just_under_2p31": check(2**31 - 100000),
    "note": "Arithmetic-only replication of the kernel's start/idx computation (no loads) with int64 stores to detect int32 wraparound at the K~2^31 boundary; actual 2^31-element tensor allocation (2x8GB+) is infeasible in this environment.",
}
print(json.dumps(res))
