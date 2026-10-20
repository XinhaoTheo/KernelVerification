import torch, sys
sys.path.insert(0, '/root/cases/case_168')
from kernel import run, reference

# Ground-truth S-box via independent GF(2^8) inverse (log/antilog tables)
def gt_sbox():
    exp=[0]*512; log=[0]*256
    x=1
    for i in range(255):
        exp[i]=x; log[x]=i
        x ^= (x<<1) ^ (0x11b if x & 0x80 else 0)
        x &= 0xff
    for i in range(255,512): exp[i]=exp[i-255]
    sb=[0]*256
    for b in range(256):
        inv = exp[255-log[b]] if b else 0
        y = inv ^ 0x63
        r = y
        for _ in range(4):
            y = ((y<<1)|(y>>7))&0xff; r ^= y
        sb[b]=r
    return sb

SB = gt_sbox()
ref = torch.tensor(SB, dtype=torch.uint8)

fails = []
# all 256 byte values
x = torch.arange(256, dtype=torch.uint8, device='cuda')
y = run(x)
if not torch.equal(y, ref):
    bad = (y != ref).nonzero().flatten().tolist()
    fails.append(('all_values', bad[:10]))

# spot examples
assert SB[0]==0x63 and SB[0x53]==0xed, (hex(SB[0]), hex(SB[0x53]))

# assorted lengths incl. edge 1, 131072, non-multiples of 256
import random
random.seed(0)
lens = [1,2,255,256,257,511,1027,4096,131071,131072]
for L in lens:
    t = torch.randint(0,256,(L,),dtype=torch.uint8,device='cuda')
    y = run(t)
    e = torch.tensor([SB[b] for b in t.tolist()],dtype=torch.uint8,device='cuda')
    if not torch.equal(y,e):
        idx = (y!=e).nonzero().flatten().tolist()
        fails.append((L, idx[:5], y[idx[:3]].tolist(), e[idx[:3]].tolist()))
    if not y.is_contiguous() or y.shape != t.shape or y.dtype != torch.uint8 or y.device != t.device:
        fails.append((L,'meta'))

# storage offset view
storage = torch.randint(0,256,(600,),dtype=torch.uint8,device='cuda')
before = storage.clone()
view = storage[137:137+259]
y = run(view)
e = torch.tensor([SB[b] for b in view.tolist()],dtype=torch.uint8,device='cuda')
if not torch.equal(y,e): fails.append(('offset_view', (y!=e).nonzero().flatten()[:5].tolist()))
if not torch.equal(storage, before): fails.append(('offset_view','input mutated'))

# fresh storage: output must not share with input
t = torch.randint(0,256,(100,),dtype=torch.uint8,device='cuda')
snap = t.clone()
y = run(t)
y[0] ^= 0xff
if not torch.equal(t, snap): fails.append(('alias','output aliases input'))

# also compare against kernel's own reference on sample
r = reference(x.cpu().cuda())
if not torch.equal(r, ref): fails.append(('reference_disagrees', (r!=ref).nonzero().flatten()[:5].tolist()))

import json
print(json.dumps({'metric':'exact match vs ground-truth S-box, all 256 values + lengths ' + str(lens) + ' + offset view + alias/mutation checks','fails':fails}))
