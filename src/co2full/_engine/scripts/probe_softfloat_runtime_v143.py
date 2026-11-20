from co2full.runtime import asset_root as _cf_asset_root

import hashlib

ROOT = _cf_asset_root()

from subtasks.softfloat_runtime_v143 import Machine, characterize


def one(job):
    image = ROOT / job["image"]
    raw = image.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == job["image_sha256"]
    machine = Machine(raw)
    outputs = [characterize(machine, int(address, 16)) for address in job["callees"]]
    return dict(binary=job["binary"], image_sha256=job["image_sha256"], helpers=outputs)
