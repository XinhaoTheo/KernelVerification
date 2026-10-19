import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_107/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
g = torch.Generator("cpu").manual_seed(1)
B,L,D = 2,16,8
st = torch.randn((B,L,D), generator=g).to("cuda")
events_seq = [torch.randn((B,Kk,D), generator=g).to("cuda") for Kk in (3,2,5)]
ptr_before = st.data_ptr()
ret = k.run(st, events_seq[0])
torch.cuda.synchronize()
same_ptr = (ret.data_ptr() == ptr_before) and (st.data_ptr() == ptr_before) and (ret is st or ret.data_ptr() == st.data_ptr())
ev_snap = [e.clone() for e in events_seq]
# chained calls on mutated state, tracking reference manually
cur = st.clone()
ok_chain = True
refs = [None]
for i, ev in enumerate(events_seq):
    e_snap = ev.clone()
    r = k.reference(cur.clone(), e_snap)
    ret = k.run(st, ev)
    torch.cuda.synchronize()
    refs.append((bool(torch.equal(st.cpu().view(torch.int32), r.cpu().view(torch.int32))),
                 bool(torch.equal(ev.cpu().view(torch.int32), e_snap.cpu().view(torch.int32)))))
    cur = r.clone()
events_untouched = all(x[1] for x in refs[1:])
data_ptr_kept = same_ptr
print(json.dumps({
  "data_ptr_preserved": data_ptr_kept,
  "events_preserved_bitwise_each_call": events_untouched,
  "chain_results": refs[1:],
}))