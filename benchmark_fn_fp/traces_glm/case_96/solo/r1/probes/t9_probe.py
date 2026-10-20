import torch, sys, json
sys.path.insert(0, '/root/cases/case_168')
from kernel import run, reference

def xtime(a):
    a <<= 1
    if a & 0x100:
        a ^= 0x11b
    return a & 0xff

def mul(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        a = xtime(a); b >>= 1
    return r

def gf_pow(a, e):
    r = 1
    while e:
        if e & 1: r = mul(r, a)
        a = mul(a, a); e >>= 1
    return r

def gt_sbox():
    sb = [0]*256
    for b in range(256):
        v = gf_pow(b, 254) if b else 0
        r = 0
        for i in range(8):
            bit = ((v>>i)&1) ^ ((v>>((i+4)%8))&1) ^ ((v>>((i+5)%8))&1) ^ ((v>>((i+6)%8))&1) ^ ((v>>((i+7)%8))&1) ^ ((0x63>>i)&1)
            r |= bit << i
        sb[b] = r
    return sb

SB = gt_sbox()
anchors_ok = (SB[0]==0x63 and SB[0x53]==0xed and SB[1]==0x7c and SB[0xff]==0x16)
if not anchors_ok:
    print(json.dumps({'error':'ground truth failed anchors', 'SB0':hex(SB[0]),'SB53':hex(SB[0x53]),'SB1':hex(SB[1]),'SBff':hex(SB[0xff])})); sys.exit(1)

ref = torch.tensor(SB, dtype=torch.uint8, device='cuda')
fails = []

x = torch.arange(256, dtype=torch.uint8, device='cuda')
y = run(x)
if not torch.equal(y, ref):
    bad = (y != ref).nonzero().flatten().tolist()
    fails.append(['all_values', len(bad), bad[:10]])

lens = [1,2,3,255,256,257,511,1027,4096,65536,131071,131072]
for L in lens:
    t = torch.randint(0,256,(L,),dtype=torch.uint8,device='cuda')
    y = run(t)
    e = ref[t.long()]
    if not torch.equal(y,e):
        idx = (y!=e).nonzero().flatten().tolist()
        fails.append([L, len(idx), idx[:5], y[idx[:3]].tolist(), [SB[t[i].item()] for i in idx[:3]]])
    if not y.is_contiguous() or tuple(y.shape)!=(L,) or y.dtype != torch.uint8 or y.device != t.device:
        fails.append([L,'meta'])

storage = torch.randint(0,256,(600,),dtype=torch.uint8,device='cuda')
before = storage.clone()
view = storage[137:137+259]
y = run(view)
if not torch.equal(y, ref[view.long()]): fails.append(['offset_view', (y!=ref[view.long()]).nonzero().flatten()[:5].tolist()])
if not torch.equal(storage, before): fails.append(['offset_view','input backing storage mutated'])

t = torch.randint(0,256,(100,),dtype=torch.uint8,device='cuda')
snap = t.clone()
y = run(t)
y[0] ^= 0xff
if not torch.equal(t, snap): fails.append(['alias','output shares storage with input'])

r = reference(x)
if not torch.equal(r, ref): fails.append(['kernel_reference_disagrees', (r!=ref).nonzero().flatten()[:5].tolist()])

# make_inputs example path
from kernel import make_inputs
v = make_inputs()[0]
if not torch.equal(run(v), ref[v.long()]): fails.append(['make_inputs', (run(v)!=ref[v.long()]).sum().item()])

print(json.dumps({'metric':'exact torch.equal vs ground-truth AES S-box (verified anchors S[0]=0x63,S[0x53]=0xed,S[1]=0x7c,S[0xff]=0x16)',
 'anchors_ok':anchors_ok,'lengths_tested':lens,'fails':fails}))
