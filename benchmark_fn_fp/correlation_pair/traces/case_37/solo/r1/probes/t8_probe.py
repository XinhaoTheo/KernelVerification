
import torch, numpy as np, json, sys
sys.path.insert(0,"/root/cases/case_b")
import kernel as K
Es=[]; finite=True
for t in range(5):
    x,a,b = K.make_inputs("cuda")
    out = K.run(x,a,b)
    dev = str(out.device)
    o = out.double().cpu().numpy()
    finite = finite and bool(np.isfinite(o).all())
    xn=x.double().cpu().numpy(); an=a.double().cpu().numpy(); bn=b.double().cpu().numpy()
    ref = an@xn + bn@xn
    den = max(np.linalg.norm(ref), 0.001*np.sqrt(64))
    Es.append(float(np.linalg.norm(o-ref)/den))
print(json.dumps({"metric":"E = ||out-ref||_2/max(||ref||_2,0.001*sqrt(64)) per contract",
 "E_runs":Es,"E_max":max(Es),"threshold":0.1,"all_finite":finite,
 "device":dev,"gpu_name":torch.cuda.get_device_name(0)}))
