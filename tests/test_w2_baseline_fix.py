"""W2 smoke test: teacher baseline fix + novelty probe + hop decomposition."""
import json, sys, os
sys.path.insert(0, "/workspace/v3/experiments")
sys.path.insert(0, "/workspace/apcs")
from phase0_g0 import run_g0

def main():
    # Select 4 hop-spanning samples from test.json
    test = json.load(open("/workspace/v3/data/test.json"))
    by_hop = {}
    for s in test:
        by_hop.setdefault(s["hop"], []).append(s)
    eval_override = [by_hop[h][0] for h in [1, 2, 3, 4]]
    
    # Run small G0 with probe
    report = run_g0(
        seed=0, n_calib=6, n_eval=4,
        output="/workspace/v3/reports/w2_baseline_fix.json",
        eval_override=eval_override, run_probe=True,
    )
    
    # Assertions
    tf = report["summary_ll"]["teacher_full"]["mean"]
    sf = report["summary_ll"]["student_full"]["mean"]
    print(f"teacher_full={tf:.2f}, student_full={sf:.2f}")
    assert tf > sf, f"FAIL: teacher_full {tf} <= student_full {sf}"
    
    # Novelty probe: both near random (em < 0.5)
    probe = report["novelty_probe"]
    assert probe["teacher"]["em"] < 0.5, f"teacher probe em {probe['teacher']['em']} too high"
    assert probe["student"]["em"] < 0.5, f"student probe em {probe['student']['em']} too high"
    
    # Hop decomposition present
    assert "summary_hop" in report
    assert set(report["summary_hop"].keys()) == {"1", "2", "3", "4"}
    
    print("ALL W2 ASSERTIONS PASSED")
    print(f"Report saved to /workspace/v3/reports/w2_baseline_fix.json")

if __name__ == "__main__":
    main()
