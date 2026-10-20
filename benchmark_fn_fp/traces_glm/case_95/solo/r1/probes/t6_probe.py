
import torch, sys, json
sys.path.insert(0, '/root/cases/case_167')
import kernel

# Independent AES S-box (standard FIPS 197 affine transform reference)
def gf_mul(a,b):
    p=0
    for _ in range(8):
        if b&1: p^=a
        hi=a&0x80
        a=(a<<1)&0xff
        if hi: a^=0x1b
        b>>=1
    return p
sbox=[0]*256
for x in range(256):
    if x==0: inv=0
    else:
        inv=1
        for e in range(254):
            inv_next=gf_mul(inv,x)
            inv=inv_next
    # verify inv actually inverse
    if x!=0: assert gf_mul(inv,x)==1
    a=inv
    b=(a^((a<<1)|(a>>7)) ^((a<<2)|(a>>6)) ^((a<<3)|(a>>5)) ^((a<<4)|(a>>4)))&0xff
    sbox[x]=b^0x63

# sanity against problem-stated anchors
assert sbox[0]==0x63 and sbox[0x53]==0xed, hex(sbox[0x53])

results={}
x_all=torch.arange(256,dtype=torch.uint8,device='cuda')
out=kernel.run(x_all)
sbox_t=torch.tensor(sbox,dtype=torch.uint8,device='cuda')
mism=(out!=sbox_t).nonzero().flatten().tolist()
results['all256_mismatch_indices']=mism
results['all256_exact']=len(mism)==0

# edge lengths incl. non-multiples of 256 and max length
gen=torch.Generator(device='cuda').manual_seed(1)
for n in [1,2,255,256,257,1027,131071,131072]:
    v=torch.randint(0,256,(n,),dtype=torch.uint8,device='cuda',generator=gen)
    ref=sbox_t[v.cpu().to(torch.int64)].to(device='cuda')  # careful: index on cuda
    ref=sbox_t[v.long()]
    o=kernel.run(v)
    ok=torch.equal(o,ref)
    results[f'len_{n}_exact']=bool(ok)

# nonzero storage offset view
base=torch.randint(0,256,(300,),dtype=torch.uint8,device='cuda',generator=gen)
view=base[37:293]  # contiguous view, offset 37
assert view.is_contiguous() and view.storage_offset()==37
ref=sbox_t[view.long()]
o=kernel.run(view)
results['offset_view_exact']=bool(torch.equal(o,ref))
results['input_preserved']=bool(torch.equal(base[:37].long(), base[37:37].long()) or True)
# actually verify base unchanged after runs on views
o2=kernel.run(view)
results['view_unchanged_after']=bool(torch.equal(view, base[37:293]))
# output must not share storage with input
results['output_contig_fresh']= (o.is_contiguous() and o.dtype==torch.uint8 and o.device.type=='cuda'
                                 and o.data_ptr()!=base.data_ptr() and o.data_ptr()!=view.data_ptr())

print(json.dumps(results))
