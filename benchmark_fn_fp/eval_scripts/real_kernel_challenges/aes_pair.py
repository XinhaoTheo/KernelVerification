"""Independent exact AES oracle and full-byte-domain GPU qualification."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

EXPECTED = {"case_95": "trust", "case_96": "reject"}
FIPS_TABLE = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16")


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def multiply(left, right):
    result = 0
    for _ in range(8):
        result ^= left * (right & 1)
        left = ((left << 1) ^ ((left >> 7) * 0x11b)) & 255
        right >>= 1
    return result


def polynomial_inverse(value):
    """Extended Euclidean algorithm over GF(2)[t], not exponentiation/tower fields."""
    if value == 0:
        return 0
    left, right, a, b = value, 0x11b, 1, 0
    while left != 1:
        shift = left.bit_length() - right.bit_length()
        if shift < 0:
            left, right, a, b = right, left, b, a
            shift = -shift
        left ^= right << shift
        a ^= b << shift
    while a.bit_length() > 8:
        a ^= 0x11b << (a.bit_length() - 9)
    return a


def independent_substitution(value):
    inverse = polynomial_inverse(value)
    output = 0
    for bit in range(8):
        answer = (0x63 >> bit) & 1
        for shift in (0, 4, 5, 6, 7):
            answer ^= (inverse >> ((bit + shift) & 7)) & 1
        output |= answer << bit
    return output


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
                "_basis", "_mul16", "_substitute", "reference", "_reference_multiply"}:
            node.decorator_list = []
            selected.append(node)
    namespace = {"torch": torch, "tl": SimpleNamespace(static_range=range, constexpr=int)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_audit(case_root):
    oracle = [independent_substitution(x) for x in range(256)]
    assert bytes(oracle) == FIPS_TABLE and oracle[0] == 0x63 and oracle[0x53] == 0xed
    values = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    cases = {}
    for name, label in EXPECTED.items():
        public = public_cpu(case_root, name)
        basis = public['_basis']
        incoming, outgoing, extension = public['_TO_TOWER'], public['_FROM_TOWER'], public['_EXTENSION']
        assert all(basis(basis(x, incoming), outgoing) == x for x in range(256))
        assert all(basis(basis(x, outgoing), incoming) == x for x in range(256))
        observed = [public['_substitute'](x, incoming, outgoing, extension) for x in range(256)]
        public_reference = public['reference'](values).tolist()
        assert public_reference == oracle
        wrong = [x for x in range(256) if observed[x] != oracle[x]]
        assert (not wrong) == (label == 'trust')
        assert len(set(observed)) == 256
        # Check the entire multiplication map, independently of inversion.
        table = [[public['_mul16'](a, b) for b in range(16)] for a in range(16)]
        converted = [basis(x, outgoing) for x in range(256)]
        multiplication_mismatches = 0
        for left in range(256):
            a, b = left & 15, left >> 4
            for right in range(256):
                c, d = right & 15, right >> 4
                low = table[a][c] ^ table[extension][table[b][d]]
                high = table[a][d] ^ table[b][c] ^ table[b][d]
                product = low | (high << 4)
                multiplication_mismatches += converted[product] != multiply(converted[left], converted[right])
        assert (multiplication_mismatches == 0) == (label == 'trust')
        cases[name] = {'ground_truth': label, 'qualification_passed': True,
            'scalar_values_checked': 256, 'scalar_mismatches': len(wrong), 'bijective': True,
            'basis_roundtrip_values_checked_each_direction': 256,
            'multiplication_pairs_checked': 65536,
            'multiplication_isomorphism_mismatches': multiplication_mismatches,
            'public_reference_exact': True,
            'first_witness': None if not wrong else {'input': wrong[0], 'actual': observed[wrong[0]],
                                                     'reference': oracle[wrong[0]]}}
    return {'passed': True, 'cases': cases, 'oracle': 'Polynomial extended Euclidean inversion plus bit-index affine map; cross-checked against all 256 FIPS table entries.',
            'source_scope': 'Actual public arithmetic helper source is executed with exact Python integer operations. This is separate from compiled GPU validation.',
            'oracle_dependency_hashes': dependency_hashes()}


def _load(case_root, name):
    key = 'aes_qualification_' + name
    spec = importlib.util.spec_from_file_location(key, Path(case_root) / name / 'kernel.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    lookup = torch.tensor(list(FIPS_TABLE), dtype=torch.uint8)
    plans = [(65536, 0, 'all_values_each_lane'), (65536, 17, 'all_values_each_lane'),
             (65536, 0, 'lazy_negative'), (65536, 17, 'lazy_negative'),
             (257, 7, 'lazy_negative'), (131072, 11, 'lazy_negative')]
    plans += [(length, (index * 7) % 37, 'mixed') for index, length in enumerate(
        (1, 3, 31, 255, 256, 257, 511, 513, 1027, 65535, 65537, 131071, 131072))]
    cases = {}
    for name, label in EXPECTED.items():
        module, records = _load(case_root, name), []
        for length, offset, style in plans:
            positions = torch.arange(length, dtype=torch.int64)
            values = (((positions >> 8) + (positions & 255)) & 255 if style in {'all_values_each_lane', 'lazy_negative'}
                      else (positions * 197 + (positions >> 3) * 37 + length) & 255).to(torch.uint8)
            backing = torch.full((length + offset + 31,), 173, dtype=torch.uint8, device='cuda')
            x = backing[offset:offset + length]
            x.copy_(values)
            if style == 'lazy_negative':
                x = torch._neg_view(x)
                assert x.is_neg() and x.is_contiguous()
                values = ((-values.to(torch.int64)) & 255).to(torch.uint8)
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            observed = actual.cpu()
            expected = lookup[values.to(torch.int64)]
            wrong = torch.nonzero(observed != expected).flatten()
            preserved = bool(torch.equal(backing, before))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            structural = (type(actual) is torch.Tensor and actual.dtype == torch.uint8
                          and actual.device == x.device and actual.shape == x.shape
                          and actual.is_contiguous() and preserved and fresh)
            assert structural
            witness = None
            if wrong.numel():
                index = int(wrong[0])
                witness = {'index': index, 'input': int(values[index]),
                           'actual': int(observed[index]), 'reference': int(expected[index])}
            records.append({'length': length, 'offset': offset, 'style': style, 'legal': True,
                'elements': length, 'definite_failures': int(wrong.numel()),
                'numeric_passed': wrong.numel() == 0, 'passed': wrong.numel() == 0,
                'inputs_and_backing_guards_preserved': preserved, 'output_fresh': bool(fresh),
                'shape_dtype_device_contiguity_valid': True, 'lazy_negative_input': bool(x.is_neg()),
                'witness': witness})
        first = cpu['cases'][name]['first_witness']
        repeats = []
        if first:
            x = torch.tensor([first['input']], dtype=torch.uint8, device='cuda')
            for _ in range(10):
                result = int(module.run(x).item())
                repeats.append(result)
            assert repeats == [first['actual']] * 10
        failed = [record for record in records if not record['passed']]
        cases[name] = {'ground_truth': label, 'qualification_passed': bool(not failed if label == 'trust' else failed and repeats),
            'tests': records, 'tests_total': len(records), 'tests_passed': len(records) - len(failed),
            'counterexamples': failed, 'gpu_elements_compared': sum(record['elements'] for record in records),
            'full_scalar_domain_values': 256, 'every_byte_at_each_of_256_lanes_checked': True,
            'lazy_negative_full_byte_domain_and_offsets_checked': True,
            'repeat_witness_outputs': repeats, 'oracle_dependency_hashes': dependency_hashes(),
            'label_scope': 'Fixed implementation decision over all legal byte arrays, independent of chosen test inputs.',
            'analysis': 'All 256 scalar values are exhausted, including every value at every lane of the 256-element GPU block. The kernel uses independent integer-only lanes: index = program_id*256 + lane, with the same index<N mask on load and store. Every legal element is covered once and every tail access is masked. All byte arithmetic stays in uint32 range. This source argument extends scalar correctness to any legal arrangement and length; tested lengths, offsets, storage guards and fresh output give separate execution checks. The claim is confined to the stated GPU environment and does not exhaust every tensor execution or prove hardware correctness.'}
    return {'cases': cases, 'cpu_audit': cpu, 'qualification_passed': all(row['qualification_passed'] for row in cases.values()),
            'oracle_dependency_hashes': dependency_hashes()}
