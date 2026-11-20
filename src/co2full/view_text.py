"""Identifier normalization for decompiled pseudocode."""

import re

KEEP = set(
    "auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while bool true false NULL nullptr size_t ssize_t wchar_t uint uintptr_t intptr_t ulong ushort uchar byte word dword qword code undefined ulonglong longlong float10 float16 __cdecl __fastcall __stdcall __thiscall __noreturn __int8 __int16 __int32 __int64 __int128 _BYTE _WORD _DWORD _QWORD _OWORD LOBYTE HIBYTE LOWORD HIWORD LODWORD HIDWORD BYTE1 BYTE2 BYTE3 SLOBYTE SHIBYTE SLOWORD SHIWORD SLODWORD SHIDWORD COERCE_FLOAT COERCE_DOUBLE COERCE_UNSIGNED_INT COERCE_UNSIGNED_INT64".split()
)
INTRINSIC = re.compile(
    r"^(?:undefined\d+|u?int\d+_t|float\d+|CONCAT\d+|SUB\d+|[ZS]EXT\d+|[S]?CARRY\d+|SBORROW\d+|POPCOUNT|LZCOUNT|NAN|ABS|SQRT)$"
)
TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\b0[xX][0-9a-fA-F]+[uUlL]*|\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[uUlLfF]*|\b[A-Za-z_]\w*',
    re.S,
)


def anonymize(code):
    mapping = {}

    def replace(m):
        t = m.group()
        if t.startswith(("//", "/*")):
            return " "
        if t.startswith(('"', "'")) or t[0].isdigit():
            return t
        if t in KEEP or INTRINSIC.fullmatch(t):
            return t
        if t not in mapping:
            mapping[t] = "v" + str(len(mapping))
        return mapping[t]

    return TOKEN.sub(replace, code)
