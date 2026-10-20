import torch, sys, json
sys.path.insert(0, '/root/cases/case_168')
from kernel import run, reference

def xtime(a):
    a <<= 1
    if a & 0x100:
        a ^= 0x11b
    return a & 0xff

def gt_sbox():
    # multiplication in GF(2^8)
    def mul(a,b):
        r=0
        while b:
            if b&1: r^=a
            a=xtime(a); b>>=1
        return r
    # inverse by brute force
    inv=[0]*256
    for a in range(1,256):
        for b in range(1,256):
            if mul(a,b)==1: inv[a]=b; break
    sb=[0]*256
    for b in range(256):
        v=inv[b]
        r=0x63
        for i in range(8):
            bit=((v>>i)&1)^((v>>((i+4)%8))&1)^((v>>((i+5)%8))&1)^((v>>((i+6)%8))&1)^((v>>((i+7)%8))&1)^((0x63>>i)&1)
            r|=bit<<i
        sb[b]=r
    return sb

SB = gt_sbox()
assert SB[0]==0x63 and SB[0x53]==0xed, (hex(SB[0]), hex(SB[0x53]))
ref = torch.tensor(SB, dtype=torch.uint8, device='cuda')

fails = []
x = torch.arange(256, dtype=torch.uint8, device='cuda')
y = run(x)
if not torch.equal(y, ref):
    bad = (y != ref).nonzero().flatten().tolist()
    fails.append(('all_values', bad[:10], y[bad[:3]].tolist(), [SB[b] for b in bad[:3]]))

lens = [1,2,255,256,257,511,1027,4096,131071,131072]
for L in lens:
    t = torch.randint(0,256,(L,),dtype=torch.uint8,device='cuda')
    y = run(t)
    e = torch.tensor([SB[b] for b in t.tolist()],dtype=torch.uint8,device='cuda')
    if not torch.equal(y,e):
        idx = (y!=e).nonzero().flatten().tolist()
        fails.append((L, idx[:5], y[idx[:3]].tolist(), [SB[t[i].item()] for i in idx[:3]]))
    if not y.is_contiguous() or tuple(y.shape)!=(L,) or y.dtype != torch.uint8 or y.device != t.device:
        fails.append((L,'meta'))

storage = torch.randint(0,256,(600,),dtype=torch.uint8,device='cuda')
before = storage.clone()
view = storage[137:137+259]
y = run(view)
e = torch.tensor([SB[b] for b in view.tolist()],dtype=torch.uint8,device='cuda')
if not torch.equal(y,e): fails.append(('offset_view', (y!=e).nonzero().flatten()[:5].tolist()))
if not torch.equal(storage, before): fails.append(('offset_view','input mutated'))

t = torch.randint(0,256,(100,),dtype=torch.uint8,device='cuda')
snap = t.clone()
y = run(t)
y[0] ^= 0xff
if not torch.equal(t, snap): fails.append(('alias','output aliases input'))

r = reference(x)
if not torch.equal(r, ref): fails.append(('reference_disagrees', (r!=ref).nonzero().flatten()[:5].tolist()))

print(json.dumps({'metric':'exact torch.equal vs brute-force ground-truth AES S-box; all 256 values, lengths, offset view, mutation/alias checks','SB[0]':hex(SB[0]),'SB[0x53]':hex(SB[0x53]),'n_lengths_tested':len(lens),'fails':fails}))
