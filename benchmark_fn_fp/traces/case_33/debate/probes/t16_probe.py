
import subprocess, sys, json, os

script = r'''
import importlib.util, json, numpy as np, torch
spec = importlib.util.spec_from_file_location("kk", "/root/cases/case_33/kernel.py")
kk = importlib.util.module_from_spec(spec); spec.loader.exec_module(kk)
dev='cuda'; rng=np.random.default_rng(2)
M,N,K = 32,64,16
gs = 32
G = -(-K//gs)   # 1 contract row
q = rng.integers(0,16,size=(K,N)).astype(np.uint32)
rows=-(-K//8)   # 2
pk=np.zeros((rows,N),dtype=np.uint32)
for k in range(K): pk[k//8] |= (q[k]&15) << (4*(k%8))
zqrow=np.full(N,4,dtype=np.uint32)
zprow=np.zeros(N//8,dtype=np.uint32)
for n in range(N): zprow[n//8] |= (zqrow[n]&15) << (4*(n%8))
zp=np.ascontiguousarray(zprow[None,:])
sc=np.full((G,N),3.0,dtype=np.float32)
a_np=rng.standard_normal((M,K)).astype(np.float32)
deq=(q.astype(np.float64)-(zqrow[None,:].astype(np.float64)+1.0))*3.0
ref=a_np.astype(np.float64)@deq
scale_ref=float(np.abs(ref).max())

res={"shape":[M,N,K],"group_size":gs,"contract_group_rows":G,
     "wrapper_num_groups":K//gs,"wrapper_clamp_max":K//gs-1,
     "ref_max_abs":scale_ref}

# dirty the memory region likely to sit before the scales/zeros allocations
junk=torch.full((4096,),-12345.0,device=dev,dtype=torch.float32); del junk
junk2=torch.full((4096,),0x7ABCDEF,device=dev,dtype=torch.int32); del junk2
torch.cuda.synchronize()

a=torch.from_numpy(a_np).to(dev)
b=torch.from_numpy(np.ascontiguousarray(pk.view(np.int32))).to(dev)
scales=torch.from_numpy(sc).to(dev)
zw=torch.from_numpy(np.ascontiguousarray(zp.view(np.int32))).to(dev)

# derived g_idx exactly as the wrapper does, for the record
gid=torch.clamp(torch.arange(K,dtype=torch.int32)//gs, max=K//gs-1)
res["wrapper_g_idx_unique"]=sorted(set(int(x) for x in gid.tolist()))

try:
    c=kk.gptq_matmul(a,b,scales,zw,gs,4); torch.cuda.synchronize()
    cn=c.double().cpu().numpy()
    res["buggy_cuda_error"]=None
    res["buggy_max_abs_err_vs_ref"]=float(np.abs(cn-ref).max())
    res["buggy_rel_err_vs_ref"]=float(np.abs(cn-ref).max()/scale_ref)
    res["buggy_max_abs_value"]=float(np.abs(cn).max())
    res["buggy_nonfinite_count"]=int((~np.isfinite(cn)).sum())
    res["buggy_all_zero"]=bool(np.all(cn==0))
except Exception as e:
    res["buggy_cuda_error"]=repr(e)[:400]
    print(json.dumps(res)); raise SystemExit(0)

# control: gs=16 -> num_groups=1, clamp max 0, single valid row used
try:
    cc=kk.gptq_matmul(a,b,scales,zw,16,4); torch.cuda.synchronize()
    ccn=cc.double().cpu().numpy()
    res["control_cuda_error"]=None
    res["control_max_abs_err_vs_ref"]=float(np.abs(ccn-ref).max())
    res["control_rel_err_vs_ref"]=float(np.abs(ccn-ref).max()/scale_ref)
    res["buggy_vs_control_max_abs_diff"]=float(np.abs(cn-ccn).max())
except Exception as e:
    res["control_cuda_error"]=repr(e)[:400]
print(json.dumps(res))
'''
open('/tmp/c3_probe.py','w').write(script)
env=dict(os.environ); env["CUDA_LAUNCH_BLOCKING"]="1"
p=subprocess.run([sys.executable,'/tmp/c3_probe.py'],capture_output=True,text=True,timeout=420,env=env)
inner=None
for line in reversed([l for l in p.stdout.splitlines() if l.strip()]):
    try:
        inner=json.loads(line); break
    except Exception:
        continue
print(json.dumps({"returncode":p.returncode,"inner":inner,
                  "stderr_tail":p.stderr[-1500:],"stdout_tail":p.stdout[-500:]}))
