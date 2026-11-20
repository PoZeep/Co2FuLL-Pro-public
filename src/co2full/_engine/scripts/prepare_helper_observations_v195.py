from scripts.probe_softfloat_runtime_v143 import one as probe

from scripts.verify_softfloat_runtime_v143 import one as validate


def fresh(job):
    result = probe(job)
    v = validate(dict(job, helpers=[h for h in result["helpers"] if h["matches"]]))
    return dict(observation=result, validation=v)
