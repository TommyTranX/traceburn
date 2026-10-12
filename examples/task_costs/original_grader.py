"""Original study grader, included for inspection.

The saved observations lack final filesystem states; do not treat this as
a complete executable regrading package for the recorded runs.
"""

def check_success(task, submitted, env):
    if submitted is None:
        return False
    a = str(submitted).strip().strip(".").casefold()
    gt = str(task["answer"]).strip().casefold()
    ok = a == gt
    if not ok:
        try:
            ok = abs(float(a.replace(",", "")) - float(gt)) < 0.01
        except (ValueError, TypeError):
            ok = False
    w = task.get("write")
    if ok and w:
        got = env.fs.get(w["path"])
        ok = got is not None and got.strip() == w["content"].strip()
    return ok
