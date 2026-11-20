from co2full.runtime import asset_root as _cf_asset_root

import hashlib

import math

import operator

import random

import struct

ROOT = _cf_asset_root()

from subtasks.softfloat_runtime_v143 import Machine, bits32, bits64, fp32, words


def challenge(match):
    rng = random.Random(143731)
    abi, name = match["abi"], match["primitive"]
    rows = []
    if abi.endswith("_binary"):
        width = int(abi[2 : abi.index("_")])
        near = (
            math.nextafter(1.0, 2.0)
            if width == 64
            else struct.unpack("<f", struct.pack("<I", bits32(1.0) + 1))[0]
        )
        pairs = [
            (math.pi, math.e),
            (-math.pi, math.e),
            (1.0, near),
            (near, 1.0),
            (-near, -1.0),
            (-1.0, -near),
            (math.pi, math.pi),
        ]
        pairs += [
            (
                math.ldexp(rng.uniform(-2, 2), rng.randint(-20, 20)),
                math.ldexp(rng.uniform(0.25, 2) * (1 if rng.getrandbits(1) else -1), rng.randint(-20, 20)),
            )
            for _ in range(17)
        ]
        operations = dict(
            add=operator.add,
            subtract=operator.sub,
            multiply=operator.mul,
            divide=operator.truediv,
            equal=operator.eq,
            less=operator.lt,
            less_equal=operator.le,
            greater=operator.gt,
            greater_equal=operator.ge,
        )
        operation = name.split("_", 1)[1]
        for a, b in pairs:
            if width == 32:
                a, b = fp32(a), fp32(b)
            bits = bits64 if width == 64 else bits32
            args = words(bits(a)) + words(bits(b)) if width == 64 else [bits(a), bits(b)]
            value = operations[operation](a, b)
            expected = (
                [int(value)]
                if isinstance(value, bool)
                else words(bits64(value))
                if width == 64
                else [bits32(value)]
            )
            rows.append(dict(inputs=[a, b], input_registers=args, expected=expected))
    elif abi in ["i32_to_fp64", "u32_to_fp64"]:
        values = [
            rng.randrange(-(1 << 31), 1 << 31) if abi.startswith("i32") else rng.randrange(1 << 32)
            for _ in range(24)
        ]
        rows = [
            dict(inputs=[v], input_registers=[v & 0xFFFFFFFF], expected=words(bits64(float(v))))
            for v in values
        ]
    elif abi in ["fp64_to_i32", "fp64_to_u32"]:
        values = [
            rng.uniform(-2147483000, 2147483000) if abi.endswith("i32") else rng.uniform(0, 4294967000)
            for _ in range(24)
        ]
        rows = [
            dict(inputs=[v], input_registers=words(bits64(v)), expected=[math.trunc(v) & 0xFFFFFFFF])
            for v in values
        ]
    elif abi == "fp32_to_fp64":
        values = [fp32(math.ldexp(rng.uniform(-2, 2), rng.randint(-20, 20))) for _ in range(24)]
        rows = [dict(inputs=[v], input_registers=[bits32(v)], expected=words(bits64(v))) for v in values]
    elif abi == "fp64_to_fp32":
        values = [math.ldexp(rng.uniform(-2, 2), rng.randint(-20, 20)) for _ in range(24)]
        rows = [dict(inputs=[v], input_registers=words(bits64(v)), expected=[bits32(v)]) for v in values]
    else:
        raise AssertionError(abi)
    assert len(rows) == 24
    return rows


def one(job):
    raw = (ROOT / job["image"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == job["image_sha256"]
    machine = Machine(raw)
    results = []
    for helper in job["helpers"]:
        validations = []
        for match in helper["matches"]:
            tests = []
            for case in challenge(match):
                observed = [
                    machine.run(
                        int(helper["address"], 16), case["input_registers"], match["mode"] == "thumb", canary
                    )
                    for canary in [0, 1]
                ]
                passed = all(
                    o["ok"] and o["registers"][: len(case["expected"])] == case["expected"] for o in observed
                )
                tests.append(dict(**case, observed=observed, passed=passed))
            validations.append(dict(match=match, passed=all(t["passed"] for t in tests), tests=tests))
        valid = [v["match"] for v in validations if v["passed"]]
        results.append(
            dict(
                address=helper["address"],
                valid_matches=valid,
                validations=validations,
                usable=len(valid) == 1,
            )
        )
    return dict(binary=job["binary"], image_sha256=job["image_sha256"], helpers=results)
