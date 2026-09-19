"""W2 smoke test: teacher baseline fix + novelty probe + hop decomposition.

GPU-only historical smoke test. It is deliberately absent from the README's test
list: `tests/test_stats_utils.py` is the CPU test. This file needs torch, a CUDA
device, the Qwen3 snapshots and the legacy (v1) `data/*.json` splits, and it
skips itself when any of those is missing.

Two things to know before trusting it:

* Its output goes to `$V3_W2_OUT` (default: the system temp directory), *not* to
  `reports/w2_baseline_fix.json`. The tracked report is provenance and must not
  be silently overwritten by a smoke test.
* The W2-era assertion that the teacher's forced-LL should exceed the student's
  does not hold: the committed `reports/w2_baseline_fix.json` records
  `teacher_full` -16.92 against `student_full` -11.33 on v1 data (n=4). The
  project later established that a teacher-forced LL gap is content-agnostic, so
  that check is reported here as a diagnostic rather than asserted. The
  evaluator behind that report is also the pre-correction one, so its `em`
  fields are legacy (see `paper/audit/METRIC_CORRECTION.md`).
"""
import json
import os
import sys
import tempfile

_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOT, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")

OUT = os.environ.get("V3_W2_OUT", os.path.join(tempfile.gettempdir(),
                                               "w2_baseline_fix.json"))


def skip(reason):
    print(f"SKIP tests/test_w2_baseline_fix.py: {reason}")
    return 0


def main():
    legacy_test = os.path.join(DATA_DIR, "test.json")
    if not os.path.isfile(legacy_test):
        return skip(f"legacy v1 data is absent ({legacy_test} is git-ignored and "
                    "regenerable with `python data/build_ood.py --legacy`)")
    for name in ("Qwen--Qwen3-8B", "Qwen--Qwen3-0.6B"):
        if not os.path.isdir(os.path.join(MODELS_DIR, name, "snapshots", "master")):
            return skip(f"model snapshot missing: {os.path.join(MODELS_DIR, name)}")
    try:
        import torch
    except Exception as exc:                                  # noqa: BLE001
        return skip(f"torch is unavailable ({type(exc).__name__})")
    if not torch.cuda.is_available():
        return skip("no CUDA device")

    sys.path.insert(0, os.path.join(_ROOT, "experiments"))
    sys.path.insert(0, APCS_DIR)
    try:
        from phase0_g0 import run_g0
    except Exception as exc:                                  # noqa: BLE001
        return skip(f"phase0_g0 is not importable "
                    f"({type(exc).__name__}: {exc})")

    # Select 4 hop-spanning samples from the legacy test split.
    with open(legacy_test, encoding="utf-8") as fh:
        test = json.load(fh)
    by_hop = {}
    for s in test:
        by_hop.setdefault(s["hop"], []).append(s)
    eval_override = [by_hop[h][0] for h in [1, 2, 3, 4]]

    report = run_g0(
        seed=0, n_calib=6, n_eval=4,
        output=OUT,
        eval_override=eval_override, run_probe=True,
    )

    tf = report["summary_ll"]["teacher_full"]["mean"]
    sf = report["summary_ll"]["student_full"]["mean"]
    print(f"diagnostic only: teacher_full={tf:.2f}, student_full={sf:.2f} "
          f"(teacher>student: {tf > sf}; the W2 premise no longer holds, see "
          f"the module docstring)")

    # Novelty probe: the synthetic domain should not be answerable from priors.
    probe = report["novelty_probe"]
    assert probe["teacher"]["em"] < 0.5, f"teacher probe em {probe['teacher']['em']} too high"
    assert probe["student"]["em"] < 0.5, f"student probe em {probe['student']['em']} too high"

    # Hop decomposition present.
    assert "summary_hop" in report
    assert set(report["summary_hop"].keys()) == {"1", "2", "3", "4"}

    print("ALL W2 ASSERTIONS PASSED")
    print(f"Report saved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
