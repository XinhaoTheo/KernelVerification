import torch, sys, json
sys.path.insert(0, '/root/cases/case_160')
import kernel as K
dev='cuda'
gen=torch.Generator('cpu').manual_seed(1234)
B=512
tiles=torch.randint(-128,128,(B,6,6),generator=gen,dtype=torch.int8).to(dev)
filters=torch.randint(-128,128,(B,3,3),generator=gen,dtype=torch.int8).to(dev)
out=K.run(tiles,filters)
ref=K.reference(tiles,filters).to(dev)
mismatches=(out!=ref)
print(json.dumps({"batch":B,"mismatch_count":int(mismatches.sum()),"total":out.numel(),
"sample_kernel":out[0].tolist() if mismatches.any() else out[0].tolist(),
"sample_ref":ref[0].tolist() if mismatches.any() else ref[0].tolist(),
"max_abs_diff":int((out.int()-ref.int()).abs().max())}))
# also recompute pre-division y via torch to check divisibility by 576
def data_t(d):
    return (4*d[...,0]-5*d[...,2]+d[...,4], -4*(d[...,1]+d[...,2])+d[...,3]+d[...,4],
            4*(d[...,1]-d[...,2])-d[...,3]+d[...,4], 2*(d[...,3]-d[...,1])-d[...,2]+d[...,4],
            2*(d[...,1]-d[...,3])-d[...,2]+d[...,4], 4*d[...,1]-5*d[...,3]+d[...,5])
def filt_t(g):
    return (g[...,0], -(g[...,0]+g[...,1]+g[...,2]), -g[...,0]+g[...,1]-g[...,2],
            g[...,0]+2*g[...,1]+4*g[...,2], g[...,0]-2*g[...,1]+4*g[...,2], g[...,2])
def out_t(m):
    return (6*m[...,0]+4*(m[...,1]+m[...,2])+m[...,3]+m[...,4],
            4*(m[...,1]-m[...,2])+2*(m[...,3]-m[...,4]),
            4*(m[...,1]+m[...,2]+m[...,3]+m[...,4]),
            4*(m[...,1]-m[...,2])+8*(m[...,3]-m[...,4])+24*m[...,5])
t=tiles.to(torch.int64); f=filters.to(torch.int64)
# columns then rows for data
cols=[]
for c in range(6):
    cols.append(data_t(torch.stack([t[:,r,c] for r in range(6)],dim=-1)))
X=[[None]*6 for _ in range(6)]
for i in range(6):
    rowv=data_t(torch.stack([cols[c][i] for c in range(6)],dim=-1))
    for j in range(6):
        X[i][j]=rowv[j]
cols=[]
for c in range(3):
    cols.append(filt_t(torch.stack([f[:,r,c] for r in range(3)],dim=-1)))
W=[[None]*6 for _ in range(6)]
for i in range(6):
    rowv=filt_t(torch.stack([cols[c][i] for c in range(3)],dim=-1))
    for j in range(6):
        W[i][j]=rowv[j]
M=[[X[i][j]*W[i][j] for j in range(6)] for i in range(6)]
# first pass over columns (like kernel: u_ij from m columns), then rows
U=[[None]*4 for _ in range(6)]
for j in range(6):
    U[j]=list(out_t(torch.stack([M[i][j] for i in range(6)],dim=-1)))
Y=[[None]*4 for _ in range(4)]
for i in range(4):
    Y[i]=list(out_t(torch.stack([U[j][i] for j in range(6)],dim=-1)))
Yt=torch.stack([torch.stack(Y[i],dim=-1) for i in range(4)],dim=1)  # [B,4,4]
rem=(Yt % 576 != 0)
print(json.dumps({"y_magnitude_max":int(Yt.abs().max()),"residue_nonzero_count":int(rem.sum()),
"div_576_matches_ref":bool(torch.equal(Yt//576,ref))}))
