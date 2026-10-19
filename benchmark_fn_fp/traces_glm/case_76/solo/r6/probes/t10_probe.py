import struct, json
from decimal import Decimal, getcontext
getcontext().prec = 60

def sigmoid_dec(x):
    x = Decimal(x)
    a = abs(x)
    s = Decimal(1); term = Decimal(1)
    for n in range(1, 200):
        term = -term * a / n
        s += term
    if x >= 0:
        return Decimal(1) / (Decimal(1) + s)
    else:
        return s / (Decimal(1) + s)

cases = [0.0029296875, 0.0068359375, 0.0107421875, -0.00146484375, -0.00341796875, -0.00537109375, -0.00927734375]
kernel_bits = ["0x3802","0x3804","0x3806","0x37fe","0x37fc","0x37fa","0x37f6"]
ref_bits = ["0x3801","0x3803","0x3805","0x37ff","0x37fd","0x37fb","0x37f7"]
res = []
for x, kb, rb in zip(cases, kernel_bits, ref_bits):
    y = sigmoid_dec(x)
    correct = struct.pack("<e", float(y))
    correct_bits = "0x%04x" % struct.unpack("<H", correct)[0]
    res.append({"x": x, "exact_sigmoid_dec": str(y), "correct_rne_bits": correct_bits,
                "kernel_bits": kb, "ref_bits": rb,
                "kernel_correct": correct_bits == kb, "ref_correct": correct_bits == rb})
    print(res[-1])
print(json.dumps(res))