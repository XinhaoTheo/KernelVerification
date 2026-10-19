import importlib.util, json
spec = importlib.util.spec_from_file_location("kk", "/root/cases/case_151/kernel.py")
kk = importlib.util.module_from_spec(spec); spec.loader.exec_module(kk)
Q = 3329
for width in (64, 128):
    bank = kk._make_coefficients(width)
    LOG_N = width.bit_length() - 1
    starts = []
    # kernel fused-stage starts
    ok = True
    for stage in range(LOG_N // 2):
        half = 1 << (2 * stage)
        start = 5 * (half - 1) // 3
        starts.append(start)
        if start + 5 * half - 1 >= len(bank):
            ok = False
    final = None
    if LOG_N % 2:
        final = 5 * ((1 << (2 * (LOG_N // 2))) - 1) // 3
        if final + width // 2 - 1 >= len(bank) or final != len(bank) - width // 2:
            ok = False
    print({"width": width, "bank_len": len(bank), "stage_starts": starts,
           "final_stage_start": final, "final_expected": len(bank) - width // 2 if LOG_N % 2 else None,
           "layout_consistent": ok})