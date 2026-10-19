"""Integer Winograd F(4x4,3x3) tile cross-correlation.

Synthetic adaptation of the F(4,3) transforms in Lavin and Gray (CVPR2016).
Filter-transform denominators are moved to the output transform. This is
not an upstream kernel or a reproduction of an upstream historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _data_transform(d0, d1, d2, d3, d4, d5):
    t0 = 4 * d0 - 5 * d2 + d4
    t1 = -4 * (d1 + d2) + d3 + d4
    t2 = 4 * (d1 - d2) - d3 + d4
    t3 = 2 * (d3 - d1) - d2 + d4
    t4 = 2 * (d1 - d3) - d2 + d4
    t5 = 4 * d1 - 5 * d3 + d5
    return t0, t1, t2, t3, t4, t5


@triton.jit
def _filter_transform(g0, g1, g2):
    t0 = g0
    t1 = -(g0 + g1 + g2)
    t2 = -g0 + g1 - g2
    t3 = g0 + 2 * g1 + 4 * g2
    t4 = g0 - 2 * g1 + 4 * g2
    t5 = g2
    return t0, t1, t2, t3, t4, t5


@triton.jit
def _output_transform(m0, m1, m2, m3, m4, m5):
    t0 = 6 * m0 + 4 * (m1 + m2) + m3 + m4
    t1 = 4 * (m1 - m2) + 2 * (m3 - m4)
    t2 = 4 * (m1 + m2 + m3 + m4)
    t3 = 4 * (m1 - m2) + 8 * (m3 - m4) + 24 * m5
    return t0, t1, t2, t3


@triton.jit
def _tiles(INPUT, FILTER, OUTPUT, BATCH: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = tile < BATCH
    d00 = tl.load(INPUT + tile * 36 + 0, mask=active, other=0).to(tl.int32)
    d01 = tl.load(INPUT + tile * 36 + 1, mask=active, other=0).to(tl.int32)
    d02 = tl.load(INPUT + tile * 36 + 2, mask=active, other=0).to(tl.int32)
    d03 = tl.load(INPUT + tile * 36 + 3, mask=active, other=0).to(tl.int32)
    d04 = tl.load(INPUT + tile * 36 + 4, mask=active, other=0).to(tl.int32)
    d05 = tl.load(INPUT + tile * 36 + 5, mask=active, other=0).to(tl.int32)
    d10 = tl.load(INPUT + tile * 36 + 6, mask=active, other=0).to(tl.int32)
    d11 = tl.load(INPUT + tile * 36 + 7, mask=active, other=0).to(tl.int32)
    d12 = tl.load(INPUT + tile * 36 + 8, mask=active, other=0).to(tl.int32)
    d13 = tl.load(INPUT + tile * 36 + 9, mask=active, other=0).to(tl.int32)
    d14 = tl.load(INPUT + tile * 36 + 10, mask=active, other=0).to(tl.int32)
    d15 = tl.load(INPUT + tile * 36 + 11, mask=active, other=0).to(tl.int32)
    d20 = tl.load(INPUT + tile * 36 + 12, mask=active, other=0).to(tl.int32)
    d21 = tl.load(INPUT + tile * 36 + 13, mask=active, other=0).to(tl.int32)
    d22 = tl.load(INPUT + tile * 36 + 14, mask=active, other=0).to(tl.int32)
    d23 = tl.load(INPUT + tile * 36 + 15, mask=active, other=0).to(tl.int32)
    d24 = tl.load(INPUT + tile * 36 + 16, mask=active, other=0).to(tl.int32)
    d25 = tl.load(INPUT + tile * 36 + 17, mask=active, other=0).to(tl.int32)
    d30 = tl.load(INPUT + tile * 36 + 18, mask=active, other=0).to(tl.int32)
    d31 = tl.load(INPUT + tile * 36 + 19, mask=active, other=0).to(tl.int32)
    d32 = tl.load(INPUT + tile * 36 + 20, mask=active, other=0).to(tl.int32)
    d33 = tl.load(INPUT + tile * 36 + 21, mask=active, other=0).to(tl.int32)
    d34 = tl.load(INPUT + tile * 36 + 22, mask=active, other=0).to(tl.int32)
    d35 = tl.load(INPUT + tile * 36 + 23, mask=active, other=0).to(tl.int32)
    d40 = tl.load(INPUT + tile * 36 + 24, mask=active, other=0).to(tl.int32)
    d41 = tl.load(INPUT + tile * 36 + 25, mask=active, other=0).to(tl.int32)
    d42 = tl.load(INPUT + tile * 36 + 26, mask=active, other=0).to(tl.int32)
    d43 = tl.load(INPUT + tile * 36 + 27, mask=active, other=0).to(tl.int32)
    d44 = tl.load(INPUT + tile * 36 + 28, mask=active, other=0).to(tl.int32)
    d45 = tl.load(INPUT + tile * 36 + 29, mask=active, other=0).to(tl.int32)
    d50 = tl.load(INPUT + tile * 36 + 30, mask=active, other=0).to(tl.int32)
    d51 = tl.load(INPUT + tile * 36 + 31, mask=active, other=0).to(tl.int32)
    d52 = tl.load(INPUT + tile * 36 + 32, mask=active, other=0).to(tl.int32)
    d53 = tl.load(INPUT + tile * 36 + 33, mask=active, other=0).to(tl.int32)
    d54 = tl.load(INPUT + tile * 36 + 34, mask=active, other=0).to(tl.int32)
    d55 = tl.load(INPUT + tile * 36 + 35, mask=active, other=0).to(tl.int32)
    g00 = tl.load(FILTER + tile * 9 + 0, mask=active, other=0).to(tl.int32)
    g01 = tl.load(FILTER + tile * 9 + 1, mask=active, other=0).to(tl.int32)
    g02 = tl.load(FILTER + tile * 9 + 2, mask=active, other=0).to(tl.int32)
    g10 = tl.load(FILTER + tile * 9 + 3, mask=active, other=0).to(tl.int32)
    g11 = tl.load(FILTER + tile * 9 + 4, mask=active, other=0).to(tl.int32)
    g12 = tl.load(FILTER + tile * 9 + 5, mask=active, other=0).to(tl.int32)
    g20 = tl.load(FILTER + tile * 9 + 6, mask=active, other=0).to(tl.int32)
    g21 = tl.load(FILTER + tile * 9 + 7, mask=active, other=0).to(tl.int32)
    g22 = tl.load(FILTER + tile * 9 + 8, mask=active, other=0).to(tl.int32)

    # Transform columns, then rows. Each vector lane is one independent tile.
    v00, v10, v20, v30, v40, v50 = _data_transform(d00, d10, d20, d30, d40, d50)
    v01, v11, v21, v31, v41, v51 = _data_transform(d01, d11, d21, d31, d41, d51)
    v02, v12, v22, v32, v42, v52 = _data_transform(d02, d12, d22, d32, d42, d52)
    v03, v13, v23, v33, v43, v53 = _data_transform(d03, d13, d23, d33, d43, d53)
    v04, v14, v24, v34, v44, v54 = _data_transform(d04, d14, d24, d34, d44, d54)
    v05, v15, v25, v35, v45, v55 = _data_transform(d05, d15, d25, d35, d45, d55)
    x00, x01, x02, x03, x04, x05 = _data_transform(v00, v01, v02, v03, v04, v05)
    x10, x11, x12, x13, x14, x15 = _data_transform(v10, v11, v12, v13, v14, v15)
    x20, x21, x22, x23, x24, x25 = _data_transform(v20, v21, v22, v23, v24, v25)
    x30, x31, x32, x33, x34, x35 = _data_transform(v30, v31, v32, v33, v34, v35)
    x40, x41, x42, x43, x44, x45 = _data_transform(v40, v41, v42, v43, v44, v45)
    x50, x51, x52, x53, x54, x55 = _data_transform(v50, v51, v52, v53, v54, v55)
    h00, h10, h20, h30, h40, h50 = _filter_transform(g00, g10, g20)
    h01, h11, h21, h31, h41, h51 = _filter_transform(g01, g11, g21)
    h02, h12, h22, h32, h42, h52 = _filter_transform(g02, g12, g22)
    w00, w01, w02, w03, w04, w05 = _filter_transform(h00, h01, h02)
    w10, w11, w12, w13, w14, w15 = _filter_transform(h10, h11, h12)
    w20, w21, w22, w23, w24, w25 = _filter_transform(h20, h21, h22)
    w30, w31, w32, w33, w34, w35 = _filter_transform(h30, h31, h32)
    w40, w41, w42, w43, w44, w45 = _filter_transform(h40, h41, h42)
    w50, w51, w52, w53, w54, w55 = _filter_transform(h50, h51, h52)

    # INT64 retains exact sums in the inverse transform.
    m00 = x00.to(tl.int64) * w00.to(tl.int64)
    m01 = x01.to(tl.int64) * w01.to(tl.int64)
    m02 = x02.to(tl.int64) * w02.to(tl.int64)
    m03 = x03.to(tl.int64) * w03.to(tl.int64)
    m04 = x04.to(tl.int64) * w04.to(tl.int64)
    m05 = x05.to(tl.int64) * w05.to(tl.int64)
    m10 = x10.to(tl.int64) * w10.to(tl.int64)
    m11 = x11.to(tl.int64) * w11.to(tl.int64)
    m12 = x12.to(tl.int64) * w12.to(tl.int64)
    m13 = x13.to(tl.int64) * w13.to(tl.int64)
    m14 = x14.to(tl.int64) * w14.to(tl.int64)
    m15 = x15.to(tl.int64) * w15.to(tl.int64)
    m20 = x20.to(tl.int64) * w20.to(tl.int64)
    m21 = x21.to(tl.int64) * w21.to(tl.int64)
    m22 = x22.to(tl.int64) * w22.to(tl.int64)
    m23 = x23.to(tl.int64) * w23.to(tl.int64)
    m24 = x24.to(tl.int64) * w24.to(tl.int64)
    m25 = x25.to(tl.int64) * w25.to(tl.int64)
    m30 = x30.to(tl.int64) * w30.to(tl.int64)
    m31 = x31.to(tl.int64) * w31.to(tl.int64)
    m32 = x32.to(tl.int64) * w32.to(tl.int64)
    m33 = x33.to(tl.int64) * w33.to(tl.int64)
    m34 = x34.to(tl.int64) * w34.to(tl.int64)
    m35 = x35.to(tl.int64) * w35.to(tl.int64)
    m40 = x40.to(tl.int64) * w40.to(tl.int64)
    m41 = x41.to(tl.int64) * w41.to(tl.int64)
    m42 = x42.to(tl.int64) * w42.to(tl.int64)
    m43 = x43.to(tl.int64) * w43.to(tl.int64)
    m44 = x44.to(tl.int64) * w44.to(tl.int64)
    m45 = x45.to(tl.int64) * w45.to(tl.int64)
    m50 = x50.to(tl.int64) * w50.to(tl.int64)
    m51 = x51.to(tl.int64) * w51.to(tl.int64)
    m52 = x52.to(tl.int64) * w52.to(tl.int64)
    m53 = x53.to(tl.int64) * w53.to(tl.int64)
    m54 = x54.to(tl.int64) * w54.to(tl.int64)
    m55 = x55.to(tl.int64) * w55.to(tl.int64)
    u00, u10, u20, u30 = _output_transform(m00, m10, m20, m30, m40, m50)
    u01, u11, u21, u31 = _output_transform(m01, m11, m21, m31, m41, m51)
    u02, u12, u22, u32 = _output_transform(m02, m12, m22, m32, m42, m52)
    u03, u13, u23, u33 = _output_transform(m03, m13, m23, m33, m43, m53)
    u04, u14, u24, u34 = _output_transform(m04, m14, m24, m34, m44, m54)
    u05, u15, u25, u35 = _output_transform(m05, m15, m25, m35, m45, m55)
    y00, y01, y02, y03 = _output_transform(u00, u01, u02, u03, u04, u05)
    y10, y11, y12, y13 = _output_transform(u10, u11, u12, u13, u14, u15)
    y20, y21, y22, y23 = _output_transform(u20, u21, u22, u23, u24, u25)
    y30, y31, y32, y33 = _output_transform(u30, u31, u32, u33, u34, u35)

    # One factor of24 per spatial axis restores the output scale.
    tl.store(OUTPUT + tile * 16 + 0, (y00 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 1, (y01 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 2, (y02 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 3, (y03 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 4, (y10 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 5, (y11 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 6, (y12 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 7, (y13 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 8, (y20 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 9, (y21 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 10, (y22 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 11, (y23 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 12, (y30 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 13, (y31 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 14, (y32 // 576).to(tl.int32), mask=active)
    tl.store(OUTPUT + tile * 16 + 15, (y33 // 576).to(tl.int32), mask=active)


def validate_inputs(tiles, filters, require_cuda=True):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for value in (tiles, filters):
        require(type(value) is torch.Tensor, "ordinary torch.Tensor required")
        require(value.layout == torch.strided and value.is_contiguous(), "contiguous strided tensors required")
        require(value.dtype == torch.int8, "INT8 inputs required")
        require(not value.requires_grad and not value.is_inference(), "ordinary non-gradient buffers required")
        require(not value.is_neg() and not value.is_conj(), "logical view flags excluded")
        require(not require_cuda or value.is_cuda, "CUDA inputs required by run")
    require(tiles.device == filters.device, "same device required")
    require(tiles.ndim == 3 and tuple(tiles.shape[1:]) == (6, 6), "tiles must have shape [B,6,6]")
    batch = tiles.shape[0]
    require(1 <= batch <= 2048, "B must be 1..2048")
    require(tuple(filters.shape) == (batch, 3, 3), "filters must have shape [B,3,3]")
    return {"valid": True, "batch": batch}


def run(tiles, filters):
    validate_inputs(tiles, filters)
    batch = tiles.shape[0]
    with torch.cuda.device(tiles.device):
        output = torch.empty((batch, 4, 4), dtype=torch.int32, device=tiles.device)
        _tiles[(triton.cdiv(batch, 16),)](tiles, filters, output, BATCH=batch, BLOCK=16, num_warps=4)
    return output


def reference(tiles, filters):
    """Direct nine-term cross-correlation, without Winograd transforms."""
    validate_inputs(tiles, filters, require_cuda=False)
    x, w = tiles.to(torch.int64), filters.to(torch.int64)
    batch = tiles.shape[0]
    output = torch.zeros((batch, 4, 4), dtype=torch.int64, device=tiles.device)
    for r in range(3):
        for c in range(3):
            output += x[:, r:r+4, c:c+4] * w[:, r, c, None, None]
    return output.to(torch.int32)


def make_inputs(device="cuda", batch=37, seed=0):
    """Example inputs; the domain includes every INT8 tile and filter."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tiles = torch.randint(-128, 128, (batch, 6, 6), generator=generator, dtype=torch.int8)
    filters = torch.randint(-128, 128, (batch, 3, 3), generator=generator, dtype=torch.int8)
    return tiles.to(device), filters.to(device)
