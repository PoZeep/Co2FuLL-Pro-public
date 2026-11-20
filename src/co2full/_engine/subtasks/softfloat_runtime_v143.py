from io import BytesIO

import math

import operator

import struct

from elftools.elf.elffile import ELFFile

import unicorn as uc

from unicorn import arm_const as ar

STACK, STACK_SIZE, RETURN = 0x70000000, 0x10000, 0x7FF00000

REGS = [getattr(ar, "UC_ARM_REG_R" + str(i)) for i in range(13)]


def bits64(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def words(value):
    return [value & 0xFFFFFFFF, value >> 32]


class Machine:
    def __init__(self, raw):
        self.machine = uc.Uc(uc.UC_ARCH_ARM, uc.UC_MODE_ARM | uc.UC_MODE_LITTLE_ENDIAN)
        self.regions = []
        elf = ELFFile(BytesIO(raw))
        assert elf.elfclass == 32 and elf.little_endian and elf["e_machine"] == "EM_ARM"
        loads = [s for s in elf.iter_segments() if s["p_type"] == "PT_LOAD" and s["p_memsz"]]
        pages = {}
        for segment in loads:
            lo, size, flags = int(segment["p_vaddr"]), int(segment["p_memsz"]), int(segment["p_flags"])
            assert 0 <= lo < lo + size < STACK
            self.regions.append((lo, lo + size, flags))
            for page in range(lo & ~4095, (lo + size + 4095) & ~4095, 4096):
                pages[page] = pages.get(page, 0) | flags
        for page, flags in pages.items():
            self.machine.mem_map(page, 4096, uc.UC_PROT_READ | (uc.UC_PROT_EXEC if flags & 1 else 0))
        for segment in loads:
            self.machine.mem_write(int(segment["p_vaddr"]), segment.data())
        self.machine.mem_map(STACK, STACK_SIZE, uc.UC_PROT_READ | uc.UC_PROT_WRITE)
        self.machine.mem_map(RETURN, 4096, uc.UC_PROT_READ | uc.UC_PROT_EXEC)
        self.machine.hook_add(uc.UC_HOOK_CODE, self._code)
        self.machine.hook_add(uc.UC_HOOK_MEM_READ | uc.UC_HOOK_MEM_WRITE, self._memory)
        self.machine.hook_add(uc.UC_HOOK_INTR, self._interrupt)

    def _stop(self, reason):
        if self.failure is None:
            self.failure = reason
        self.machine.emu_stop()

    def _code(self, machine, address, size, _):
        self.instructions += 1
        if not any(lo <= address and address + size <= hi and flags & 1 for lo, hi, flags in self.regions):
            self._stop("execution_outside_original_executable_segments")

    def _memory(self, machine, access, address, size, value, _):
        if STACK <= address and address + size <= STACK + STACK_SIZE:
            offset = address - STACK
            if access == uc.UC_MEM_WRITE:
                self.initialized[offset : offset + size] = b"\x01" * size
            elif not all(self.initialized[offset : offset + size]):
                self._stop("read_of_uninitialized_stack")
            return
        if access == uc.UC_MEM_WRITE:
            self._stop("write_outside_private_stack")
        elif not any(
            lo <= address and address + size <= hi and not flags & 2 for lo, hi, flags in self.regions
        ):
            self._stop("read_outside_readonly_original_image")

    def _interrupt(self, machine, number, _):
        self._stop("guest_interrupt_" + str(number))

    def run(self, address, args, thumb=False, canary=0):
        assert len(args) <= 4
        machine = self.machine
        self.failure, self.instructions = None, 0
        self.initialized = bytearray(STACK_SIZE)
        machine.mem_write(STACK, bytes(STACK_SIZE))
        # Vary unspecified registers to detect hidden inputs.
        seed = [0x13579BDF, 0xA6C35E19][canary]
        for index, register in enumerate(REGS):
            machine.reg_write(register, (seed ^ (index * 0x9E3779B9)) & 0xFFFFFFFF)
        for index in range(32):
            machine.reg_write(getattr(ar, "UC_ARM_REG_D" + str(index)), (seed << 32) | (seed ^ index))
        machine.reg_write(ar.UC_ARM_REG_CPSR, 0x10 | (0xA0000000 if canary else 0x40000000))
        machine.reg_write(ar.UC_ARM_REG_SP, STACK + STACK_SIZE // 2)
        machine.reg_write(ar.UC_ARM_REG_LR, RETURN)
        for register, value in zip(REGS, args):
            machine.reg_write(register, value & 0xFFFFFFFF)
        try:
            machine.emu_start((address & ~1) | int(thumb), RETURN, timeout=200000, count=20000)
        except uc.UcError as error:
            self.failure = self.failure or "emulator_" + str(error)
        returned = machine.reg_read(ar.UC_ARM_REG_PC) == RETURN
        if not returned and self.failure is None:
            self.failure = "instruction_or_time_budget_without_return"
        return dict(
            ok=self.failure is None and returned,
            error=self.failure,
            registers=[machine.reg_read(r) for r in REGS[:4]],
            instructions=self.instructions,
        )


def catalogs():
    """Register conventions and test inputs for arithmetic helpers."""
    pairs = [
        (1.5, 2.25),
        (5.5, -2.0),
        (-3.0, 4.0),
        (0.0, 2.0),
        (-0.0, -2.0),
        (1.0, 1.0),
        (0.875, 0.25),
        (1e10, 3.0),
        (2.0**-20, -(2.0**-18)),
        (-7.25, -0.5),
    ]
    ops = dict(add=operator.add, subtract=operator.sub, multiply=operator.mul, divide=operator.truediv)
    comparisons = dict(
        equal=operator.eq,
        less=operator.lt,
        less_equal=operator.le,
        greater=operator.gt,
        greater_equal=operator.ge,
    )
    groups = []
    for width in [64, 32]:
        rows = []
        for a, b in pairs:
            if width == 32:
                a, b = fp32(a), fp32(b)
            bits = bits64 if width == 64 else bits32
            args = words(bits(a)) + words(bits(b)) if width == 64 else [bits(a), bits(b)]
            expected = {}
            for name, operation in ops.items():
                value = bits(operation(a, b))
                expected[f"fp{width}_{name}"] = words(value) if width == 64 else [value]
            for name, operation in comparisons.items():
                expected[f"fp{width}_{name}"] = [int(operation(a, b))]
            rows.append(dict(inputs=[a, b], registers=args, expected=expected))
        groups.append(dict(abi=f"fp{width}_binary", rows=rows))
    for signed in [True, False]:
        values = (
            [0, 1, -1, 1234567, -1048576, 2147483647, -2147483648]
            if signed
            else [0, 1, 65537, 2147483648, 4294967295, 1234567, 4000000000]
        )
        name = ("i32" if signed else "u32") + "_to_fp64"
        groups.append(
            dict(
                abi=name,
                rows=[
                    dict(inputs=[v], registers=[v & 0xFFFFFFFF], expected={name: words(bits64(float(v)))})
                    for v in values
                ],
            )
        )
    for signed in [True, False]:
        values = (
            [1.75, -2.5, 0.0, 32769.25, -1000000.875, 1000000.875, 1073741824.0]
            if signed
            else [1.75, 0.0, 32769.25, 1000000.875, 2147483679.125, 4294967295.0, 1073741824.0]
        )
        name = "fp64_to_" + ("i32" if signed else "u32")
        groups.append(
            dict(
                abi=name,
                rows=[
                    dict(
                        inputs=[v], registers=words(bits64(v)), expected={name: [math.trunc(v) & 0xFFFFFFFF]}
                    )
                    for v in values
                ],
            )
        )
    values = [1.5, -2.25, 0.0, -0.0, 1234.125, -1e10, 2.0**-20]
    groups.append(
        dict(
            abi="fp32_to_fp64",
            rows=[
                dict(
                    inputs=[fp32(v)], registers=[bits32(v)], expected={"fp32_to_fp64": words(bits64(fp32(v)))}
                )
                for v in values
            ],
        )
    )
    groups.append(
        dict(
            abi="fp64_to_fp32",
            rows=[
                dict(inputs=[v], registers=words(bits64(v)), expected={"fp64_to_fp32": [bits32(v)]})
                for v in values
            ],
        )
    )
    return groups


def characterize(machine, address):
    observations, matches = [], []
    for thumb in [False, True]:
        for group in catalogs():
            candidates = set(group["rows"][0]["expected"])
            samples = []
            for row in group["rows"]:
                outputs = [machine.run(address, row["registers"], thumb, canary) for canary in [0, 1]]
                samples.append(
                    dict(inputs=row["inputs"], input_registers=row["registers"], executions=outputs)
                )
                if not all(o["ok"] for o in outputs):
                    candidates.clear()
                    break
                candidates = {
                    name
                    for name in candidates
                    if all(
                        o["registers"][: len(row["expected"][name])] == row["expected"][name] for o in outputs
                    )
                }
                if not candidates:
                    break
            observations.append(
                dict(
                    mode="thumb" if thumb else "arm",
                    abi=group["abi"],
                    samples=samples,
                    matching_primitives=sorted(candidates),
                )
            )
            matches.extend(
                dict(
                    mode="thumb" if thumb else "arm",
                    abi=group["abi"],
                    primitive=name,
                    probes=len(samples),
                    unspecified_register_variations=2,
                )
                for name in sorted(candidates)
            )
    return dict(
        address=hex(address),
        matches=matches,
        observations=observations,
        limitation="Matches are empirical finite-input register-ABI hypotheses. No source identity, formal equivalence, full IEEE754 edge-case coverage, or caller argument mapping is established.",
    )
