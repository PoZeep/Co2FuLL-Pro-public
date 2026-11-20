from pathlib import Path

import struct


def read_elf(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b"\x7fELF" and raw[4] in (1, 2) and raw[5] in (1, 2)
    is64, endian = raw[4] == 2, "<" if raw[5] == 1 else ">"
    header = struct.unpack_from(endian + ("HHIQQQIHHHHHH" if is64 else "HHIIIIIHHHHHH"), raw, 16)
    offset, entry_size, count, names_index = header[5], header[10], header[11], header[12]
    fmt = endian + ("IIQQQQIIQQ" if is64 else "IIIIIIIIII")
    assert count > 0 and 0 <= names_index < count and entry_size >= struct.calcsize(fmt)
    sections = [struct.unpack_from(fmt, raw, offset + i * entry_size) for i in range(count)]
    names = raw[sections[names_index][4] : sections[names_index][4] + sections[names_index][5]]
    allowed = []
    for s in sections:
        name = names[s[0] :].split(b"\0", 1)[0].decode("utf-8", errors="replace")
        if s[1] == 1 and s[2] & 2 and not s[2] & 5 and name.startswith((".rodata", ".srodata")):
            assert s[4] + s[5] <= len(raw)
            allowed.append(dict(name=name, address=s[3], offset=s[4], size=s[5]))
    return dict(raw=raw, sections=allowed, endian=endian, bits=64 if is64 else 32)
