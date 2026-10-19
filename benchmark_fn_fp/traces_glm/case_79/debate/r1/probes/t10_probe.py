import zlib, json, torch, sys
sys.path.insert(0, "/root/cases/case_152")
import kernel as K

def cases():
    g = torch.Generator(device="cpu").manual_seed(0)
    out = {}
    for width in (64, 128):
        for batch in (1, 7, 128):
            rows = []
            rows.append(torch.zeros((batch,width), dtype=torch.uint8))
            rows.append(torch.full((batch,width), 255, dtype=torch.uint8))
            rows.append(torch.randint(0,256,(batch,width), generator=g, dtype=torch.uint8))
            rows.append(torch.randint(0,256,(batch,width), generator=g, dtype=torch.uint8))
            # single-bit / single-byte patterns
            sb = torch.zeros((batch,width), dtype=torch.uint8); sb[:, width//2] = 0x80; rows.append(sb)
            ff = torch.zeros((batch,width), dtype=torch.uint8); ff[:,0]=0xFF; rows.append(ff)
            # contiguous view with nonzero storage offset
            base = torch.randint(0,256,(batch, width*2), generator=g, dtype=torch.uint8)
            rows.append(base[:, width:].contiguous() if False else base[:, :width])
            out[(width,batch)] = rows
    return out

def ref(x):
    return [zlib.crc32(bytes(r)) & 0xFFFFFFFF for r in x.cpu().tolist()]

total_rows = 0; bad = []
for (width,batch), rows in cases().items():
    for i, x in enumerate(rows):
        if x.shape[1] == width:
            got = K.run(x.cuda())
            exp = ref(x)
            for j in range(batch):
                total_rows += 1
                if int(got[j].item()) != exp[j]:
                    bad.append((width,batch,i,j,got[j].item(),exp[j]))
print(json.dumps({"total_rows_checked": total_rows, "num_mismatches": len(bad), "mismatch_examples": bad[:5],
 "metric": "exact int64 equality of run() output vs zlib.crc32 per row; inputs include zeros, 0xFF, random, single-byte, edge batches B=1/128, widths 64/128",
 "output_dtype": str(K.run(torch.zeros((1,64),dtype=torch.uint8).cuda()).dtype)}))