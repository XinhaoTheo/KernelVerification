# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted 2026: standalone FP16 paged-attention interface and bounded domain.
# See LICENSE for upstream attribution and the license text.
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention(Q_PTR, K_PTR, V_PTR, TABLE, LENGTHS, OUT,
                     QT: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                     D: tl.constexpr, PAGE: tl.constexpr, TABLE_WIDTH: tl.constexpr,
                     GROUP: tl.constexpr, WINDOW: tl.constexpr,
                     SCALE: tl.constexpr, BQ: tl.constexpr,
                     BM: tl.constexpr = 16, TILE: tl.constexpr = 64):
    qb = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = tl.arange(0, BM)
    dims = tl.arange(0, D)
    tokens = tl.arange(0, TILE)
    qpos = qb * BQ + rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    qo = ((batch * QT + qpos[:, None]) * HQ + head[:, None]) * D + dims[None, :]
    q = tl.load(Q_PTR + qo, mask=qpos[:, None] < QT, other=0.0)
    length = tl.load(LENGTHS + batch)
    context = length - QT
    max_prefix = tl.minimum(context + qb * BQ + BQ, length)
    num_tiles = tl.cdiv(max_prefix, TILE)
    tile_start = 0
    tile_end = num_tiles
    if WINDOW > 0:
        first_key = context + qb * BQ - WINDOW + 1
        last_key = context + tl.minimum(qb * BQ + BQ - 1, QT - 1)
        tile_start = tl.maximum(0, first_key // TILE)
        tile_end = tl.minimum(last_key // TILE + 1, num_tiles)

    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.full((BM,), 1.0, tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    for tile in range(tile_start, tile_end):
        logical = tile * TILE + tokens
        tile_mask = logical < max_prefix
        physical = tl.load(TABLE + batch * TABLE_WIDTH + logical // PAGE,
                           mask=tile_mask, other=0).to(tl.int64)
        base = ((physical * PAGE + logical % PAGE) * HK + kv_head) * D
        k = tl.load(K_PTR + base[None, :] + dims[:, None],
                    mask=tile_mask[None, :], other=0.0)
        v = tl.load(V_PTR + base[:, None] + dims[None, :],
                    mask=tile_mask[:, None], other=0.0)
        absolute_q = context + qpos[:, None]
        allowed = logical[None, :] <= absolute_q
        if WINDOW > 0:
            allowed = allowed & ((absolute_q - logical[None, :]) < WINDOW)
        scores = SCALE * tl.dot(q, k)
        scores = tl.where((qpos[:, None] < QT) & allowed & tile_mask[None, :],
                          scores, float("-inf"))
        updated_maximum = tl.maximum(maximum, tl.max(scores, 1))
        updated_maximum = tl.where(updated_maximum > float("-inf"), updated_maximum, 0.0)
        probabilities = tl.exp(scores - updated_maximum[:, None])
        alpha = tl.exp(maximum - updated_maximum)
        accumulator = accumulator * alpha[:, None]
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        maximum = updated_maximum
        if WINDOW > 0:
            qpos_lo = qb * BQ
            v = tl.where((context + qpos_lo - logical[:, None]) < WINDOW, v, 0.0)
        accumulator += tl.dot(probabilities.to(v.dtype), v)
    result = accumulator / denominator[:, None]
    tl.store(OUT + qo, result, mask=qpos[:, None] < QT)


def validate_inputs(q, k, v, block_table, lengths, window, *, require_cuda=True):
    """Check the complete public input domain; also usable on CPU by oracles."""
    values = (q, k, v, block_table, lengths)
    assert all(type(t) is torch.Tensor for t in values)
    assert all(t.layout == torch.strided and t.is_contiguous() for t in values)
    assert all(not t.requires_grad and not t.is_inference() and
               not t.is_neg() and not t.is_conj() for t in values)
    assert all(t.device == q.device for t in values)
    if require_cuda:
        assert q.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert block_table.dtype == lengths.dtype == torch.int32
    assert q.ndim == k.ndim == v.ndim == 4 and block_table.ndim == 2 and lengths.ndim == 1
    b, qt, hq, d = q.shape
    pages, page, hk, kd = k.shape
    assert 1 <= b <= 4 and 1 <= qt <= 33 and d in (32, 64, 128)
    assert 1 <= hk <= 4 and hq % hk == 0 and hq // hk in (1, 2, 4, 8) and hq <= 16
    assert 1 <= pages <= 256 and page in (16, 32, 64) and kd == d and v.shape == k.shape
    assert lengths.shape == (b,) and block_table.shape[0] == b
    assert 1 <= block_table.shape[1] <= 512 // page
    assert type(window) is int and 0 <= window <= 256
    # Inputs own disjoint storage. No alias or framework metadata semantics are part of this interface.
    pointers = [t.untyped_storage().data_ptr() for t in values]
    assert len(set(pointers)) == len(pointers)
    assert bool(torch.isfinite(q).all()) and float(q.abs().max()) <= 1.0
    for batch in range(b):
        length = int(lengths[batch])
        assert qt <= length <= 512 and triton.cdiv(length, page) <= block_table.shape[1]
        entries = block_table[batch, :triton.cdiv(length, page)]
        assert bool(((entries >= 0) & (entries < pages)).all())
        lo = max(0, length - qt - window + 1) if window else 0
        positions = torch.arange(lo, length, device=q.device)
        mapped = block_table[batch, positions // page].long()
        for cache in (k, v):
            live = cache[mapped, positions % page]
            assert bool(torch.isfinite(live).all()) and float(live.abs().max()) <= 1.0
    return True


def run(q, k, v, block_table, lengths, window):
    """Return FP16[B,Q,Hq,D] paged causal attention; preserve every input."""
    validate_inputs(q, k, v, block_table, lengths, window)
    b, qt, hq, d = q.shape
    page, hk = k.shape[1:3]
    group = hq // hk
    bq = 16 // group
    output = torch.empty_like(q)
    _paged_attention[(triton.cdiv(qt, bq), hk, b)](
        q, k, v, block_table, lengths, output,
        QT=qt, HQ=hq, HK=hk, D=d, PAGE=page, TABLE_WIDTH=block_table.shape[1],
        GROUP=group, WINDOW=window, SCALE=1.0 / math.sqrt(d), BQ=bq,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


def reference(q, k, v, block_table, lengths, window):
    """FP64 logical-position reference, independent of tile processing."""
    validate_inputs(q, k, v, block_table, lengths, window, require_cuda=False)
    b, qt, hq, d = q.shape
    page, hk = k.shape[1:3]
    output = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    head_map = torch.arange(hq, device=q.device) // (hq // hk)
    for batch in range(b):
        context = int(lengths[batch]) - qt
        for row in range(qt):
            end = context + row + 1
            begin = max(0, end - window) if window else 0
            positions = torch.arange(begin, end, device=q.device)
            mapped = block_table[batch, positions // page].long()
            keys = k[mapped, positions % page][:, head_map].double()
            vals = v[mapped, positions % page][:, head_map].double()
            scores = (keys * q[batch, row].double()[None]).sum(-1) / math.sqrt(d)
            probabilities = torch.softmax(scores, dim=0)
            output[batch, row] = (probabilities[:, :, None] * vals).sum(0)
    return output


def error_ratio(output, inputs):
    target = reference(*inputs)
    return float(((output.double() - target).abs() / (0.003 + 0.003 * target.abs())).max())


def make_inputs(device="cuda", seed=0):
    """An ordinary example, not a restriction of the supported input domain."""
    generator = torch.Generator().manual_seed(seed)
    q = (2 * torch.rand((2, 9, 4, 64), generator=generator) - 1).half()
    k = (2 * torch.rand((16, 16, 2, 64), generator=generator) - 1).half()
    v = (2 * torch.rand(k.shape, generator=generator) - 1).half()
    table = torch.randperm(16, generator=generator).reshape(2, 8).to(torch.int32)
    lengths = torch.tensor([97, 113], dtype=torch.int32)
    return tuple(t.to(device) for t in (q, k, v, table, lengths)) + (41,)
