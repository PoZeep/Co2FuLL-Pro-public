import re

TAG = re.compile(r"(?:<|`{3})same_source>\s*(yes|no)\s*</same_source>", re.I)


def decode(out):
    if out.get("error") or out.get("finish_reason") != "stop":
        return None
    text = out.get("raw_output", "")
    if out.get("constrained"):
        return int(text.strip() == "yes") if text.strip() in ["yes", "no"] else None
    tags = [t.lower() for t in TAG.findall(text)]
    return int(tags[0] == "yes") if tags and len(set(tags)) == 1 else None
