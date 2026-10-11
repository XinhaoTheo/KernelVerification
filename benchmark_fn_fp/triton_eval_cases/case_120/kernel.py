"""Standalone GPTQ-format INT4 projection with activation-order preparation.

AutoGPTQ portions: Copyright (c) 2023 潘其威(William), MIT.
Marlin packing layout: Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0.
See LICENSE. 2026 adaptation: explicit T4-compatible Triton preparation/GEMM,
small shape bounds, GPTQ asymmetric zero points and independent public reference.
This uses the Marlin packed layout, not its Ampere-only CUDA execution kernel.
"""
import torch
import triton
import triton.language as tl


def _layout_permutation(device):
    permutation = []
    for i in range(32):
        column = i // 4
        within = []
        for block in (0, 1):
            for row in (2*(i % 4), 2*(i % 4)+1,
                        2*(i % 4+4), 2*(i % 4+4)+1):
                within.append(16*row + column + 8*block)
        for j in range(4):
            permutation.extend(p + 256*j for p in within)
    permutation = torch.tensor(permutation, device=device, dtype=torch.int32)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)
    return permutation.reshape(-1, 8)[:, interleave].reshape(-1).contiguous()


@triton.jit
def _repack(QWEIGHT, ORDER, PERM, PACKED,
            K: tl.constexpr, N: tl.constexpr, NT: tl.constexpr,
            B: tl.constexpr):
    tile_k = tl.program_id(0)
    tile_n = tl.program_id(1)
    word = tl.arange(0, B)
    packed = tl.zeros((B,), tl.uint32)
    for nibble in tl.static_range(8):
        source = tl.load(PERM + word*8 + nibble)
        local_k = (source % 256) // 16
        local_n = (source // 256)*16 + source % 16
        sorted_k = tile_k*16 + local_k
        original_k = tl.load(ORDER + sorted_k, mask=sorted_k < K, other=0)
        column = tile_n*64 + local_n
        value = tl.load(QWEIGHT + (original_k // 8)*N + column,
                        mask=(sorted_k < K) & (column < N), other=0).to(tl.uint32)
        quant = (value >> (4*(original_k % 8))) & 15
        packed |= quant << (4*nibble)
    tl.store(PACKED + (tile_k*NT + tile_n)*128 + word, packed)


@triton.jit
def _int4_gemm(X, PACKED, SCALE, QZERO, ORDER, OUT,
               M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
               NT: tl.constexpr, ZN: tl.constexpr, GROUP_SIZE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    columns = tl.program_id(1)*BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block*BK + inner
        original = tl.load(ORDER + kk, mask=kk < K, other=0)
        activation = tl.load(X + rows[:, None]*K + original[None, :],
                             mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        word = 4*(4*(columns[None, :] % 8) + (kk[:, None] % 8)//2) + (columns[None, :] % 64)//16
        lane = 4*(kk[:, None] % 2) + 2*((columns[None, :] % 16)//8) + (kk[:, None] % 16)//8
        address = ((kk[:, None]//16)*NT + columns[None, :]//64)*128 + word
        packed = tl.load(PACKED + address,
                         mask=(kk[:, None] < K) & (columns[None, :] < N), other=0).to(tl.uint32)
        quant = (packed >> (4*lane)) & 15
        group = kk // GROUP_SIZE
        scale = tl.load(SCALE + group[:, None]*N + columns[None, :],
                        mask=(kk[:, None] < K) & (columns[None, :] < N), other=0.0).to(tl.float32)
        zero_word = tl.load(QZERO + group[:, None]*ZN + columns[None, :]//8,
                            mask=(kk[:, None] < K) & (columns[None, :] < N), other=0).to(tl.uint32)
        zero = ((zero_word >> (4*(columns[None, :] % 8))) & 15) + 1
        weight = ((quant.to(tl.int32)-zero.to(tl.int32)).to(tl.float32)*scale).to(tl.float16)
        accumulator = tl.dot(activation, weight, accumulator)
    tl.store(OUT + rows[:, None]*N + columns[None, :], accumulator,
             mask=(rows[:, None] < M) & (columns[None, :] < N))


def validate_inputs(x, qweight, scales, qzeros, g_idx, group_size, require_cuda=True):
    """Enforce the full public contract; CPU validation is also available."""
    tensors = (x, qweight, scales, qzeros, g_idx)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for tensor in tensors:
        require(type(tensor) is torch.Tensor, "ordinary torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(), "contiguous strided tensors required")
        require(not tensor.requires_grad and not tensor.is_inference(), "ordinary non-gradient materialized buffers required")
        require(not tensor.is_neg() and not tensor.is_conj(), "logical view flags are outside this domain")
    require(all(t.device == x.device for t in tensors), "all inputs must share one device")
    require(not require_cuda or x.is_cuda, "CUDA inputs required by run")
    require(type(group_size) is int and group_size in (16,32,64), "group_size must be16,32 or64")
    require(x.ndim == 2 and qweight.ndim == 2 and scales.ndim == 2,
            "expected x[M,K], qweight[ceil(K/8),N], scales[G,N]")
    m,k = x.shape
    n = qweight.shape[1]
    groups = (k+group_size-1)//group_size
    require(1 <= m <= 65 and 1 <= k <= 128 and 1 <= n <= 96, "shape outside supported bounds")
    require(qweight.shape == ((k+7)//8,n) and scales.shape == (groups,n), "qweight/scales shape mismatch")
    require(qzeros.shape == (groups,(n+7)//8) and g_idx.shape == (k,), "qzeros/g_idx shape mismatch")
    require(x.dtype == scales.dtype == torch.float16, "activations/scales must be FP16")
    require(qweight.dtype == qzeros.dtype == g_idx.dtype == torch.int32, "packed buffers and g_idx must be int32")
    require(bool(torch.isfinite(x).all().item()) and bool(torch.isfinite(scales).all().item()), "floating inputs must be finite")
    require(bool((x.abs() <= .5).all().item()), "activation magnitude exceeds0.5")
    require(bool(((scales > 0) & (scales <= .03125)).all().item()), "scales must be in(0,1/32]")
    require(bool(((g_idx >= 0) & (g_idx < groups)).all().item()), "group id out of range")
    counts = torch.bincount(g_idx.long(), minlength=groups)
    expected = torch.tensor([min(group_size,k-g*group_size) for g in range(groups)],
                            device=x.device,dtype=counts.dtype)
    require(bool(torch.equal(counts,expected)), "group histogram does not match group_size and final partial group")
    intervals = [(t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()) for t in tensors]
    require(all(a1 <= b0 or b1 <= a0 for i,(a0,a1) in enumerate(intervals)
                for b0,b1 in intervals[i+1:]), "input byte ranges must be disjoint")
    return {"valid":True,"rows":m,"input_width":k,"output_width":n,
            "groups":groups,"group_size":group_size}


def run(x, qweight, scales, qzeros, g_idx, group_size):
    """Prepare the packed representation, then return a fresh FP32 matrix."""
    validate_inputs(x,qweight,scales,qzeros,g_idx,group_size)
    m,k = x.shape
    n = qweight.shape[1]
    with torch.cuda.device(x.device):
        order = torch.argsort(g_idx,stable=True).to(torch.int32)
        permutation = _layout_permutation(x.device)
        kt,nt = triton.cdiv(k,16),triton.cdiv(n,64)
        packed = torch.empty((kt*nt*128,),device=x.device,dtype=torch.int32)
        _repack[(kt,nt)](qweight,order,permutation,packed,K=k,N=n,NT=nt,B=128,num_warps=4)
        output = torch.empty((m,n),device=x.device,dtype=torch.float32)
        _int4_gemm[(triton.cdiv(m,16),triton.cdiv(n,32))](x,packed,scales,qzeros,
            order,output,M=m,K=k,N=n,NT=nt,ZN=triton.cdiv(n,8),GROUP_SIZE=group_size,
            BM=16,BN=32,BK=32,num_warps=4)
    return output


def reference(x, qweight, scales, qzeros, g_idx, group_size):
    """Direct canonical bit decode, H16 weights, independent FP64 target."""
    validate_inputs(x,qweight,scales,qzeros,g_idx,group_size,require_cuda=False)
    m,k = x.shape
    n = qweight.shape[1]
    columns = torch.arange(n,device=x.device)
    decoded = torch.empty((k,n),device=x.device,dtype=torch.float16)
    for row in range(k):
        quant = (qweight[row//8].long() >> (4*(row%8))) & 15
        group = int(g_idx[row].item())
        zero = ((qzeros[group,columns//8].long() >> (4*(columns%8))) & 15) + 1
        decoded[row] = ((quant-zero).double()*scales[group].double()).half()
    return x.double() @ decoded.double()


def make_inputs(device="cuda", rows=7, input_width=64, output_width=65, group_size=16, seed=0):
    """Convenience generator; not a restriction on the legal input domain."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    k,n = input_width,output_width
    groups = (k+group_size-1)//group_size
    x = (torch.rand((rows,k),generator=rng)-.5).half()
    codes = torch.randint(0,16,(((k+7)//8)*8,n),generator=rng,dtype=torch.int64)
    qweight = torch.zeros(((k+7)//8,n),dtype=torch.int64)
    for i in range(8):
        qweight |= codes[i::8] << (4*i)
    zero_codes = torch.randint(0,16,(groups,((n+7)//8)*8),generator=rng,dtype=torch.int64)
    qzeros = torch.zeros((groups,(n+7)//8),dtype=torch.int64)
    for i in range(8):
        qzeros |= zero_codes[:,i::8] << (4*i)
    scales = (.002 + .029*torch.rand((groups,n),generator=rng)).half()
    ids = (torch.arange(k)//group_size)[torch.randperm(k,generator=rng)].int()
    return (x.to(device),qweight.int().to(device),scales.to(device),
            qzeros.int().to(device),ids.to(device),group_size)
