import re


def rationale(raw):
    m = re.search(r"<explanation>(.*?)</explanation>", raw, re.S | re.I)
    return (m[1] if m else re.sub(r"<same_source>.*?</same_source>", "", raw, flags=re.S | re.I)).strip()
