from subtasks.compile_yes_v112 import content as previous_content


def content(row, arm):
    assert arm.rsplit("_", 1)[0] == "review"
    return previous_content(row, "control_" + arm.rsplit("_", 1)[1])
