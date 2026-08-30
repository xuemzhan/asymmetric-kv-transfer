#!/usr/bin/env python3
"""
verify_paper_numbers.py — Automated number cross-verification script.

Asserts key numbers in the paper (main.tex) against authoritative JSON truth
sources: g0_v2_summary.json and w4_cca_perhead.json.

Parts:
  A. G0 numbers (teacher_full, student_full, Affine_K-only, DeltaLL)
  B. CCA Table 3 verification (6 pairs)
  C. Forbidden words scan
  D. Output and exit code
"""

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path("/workspace/v3")
G0_JSON = PROJECT / "reports" / "g0_v2_summary.json"
CCA_JSON = PROJECT / "reports" / "w4_cca_perhead.json"
MAIN_TEX = PROJECT / "paper" / "main.tex"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
results = []  # list of (bool, str) — (passed, message)

def assert_check(passed: bool, msg: str):
    results.append((passed, msg))
    tag = "PASS" if passed else "FAIL"
    print(f"{tag} {msg}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tex(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def mean_seeds(d: dict, key: str) -> float:
    """Average summary_ll.<key>.<seed>.mean across seeds '0','1','2'."""
    seeds = d["summary_ll"][key]
    vals = [seeds[s]["mean"] for s in ("0", "1", "2")]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
g0 = load_json(G0_JSON)
cca = load_json(CCA_JSON)
tex = load_tex(MAIN_TEX)

# ===========================================================================
# Part A: G0 truth-source numbers from g0_v2_summary.json
# ===========================================================================
print("=== Part A: G0 numbers ===")

teacher_mean = mean_seeds(g0, "teacher_full")
student_mean = mean_seeds(g0, "student_full")
affine_k_mean = mean_seeds(g0, "Affine_K-only")

teacher_r2 = round(teacher_mean, 2)
student_r2 = round(student_mean, 2)
affine_k_r2 = round(affine_k_mean, 2)

teacher_r2_s = f"{teacher_r2}"      # e.g. "-14.77"
student_r2_s = f"{student_r2}"      # e.g. "-9.89"
affine_k_r2_s = f"{affine_k_r2}"   # e.g. "-7.27"

dell_r2 = round(affine_k_r2 - teacher_r2, 2)
dell_r2_s = f"{dell_r2:+.2f}"      # e.g. "+7.50"

# A1: teacher_full -14.77 present
assert_check(
    teacher_r2_s in tex,
    f"teacher_full 2dp '{teacher_r2_s}' present in main.tex (raw mean={teacher_mean:.6f})"
)
# A2: teacher_full -14.79 NOT present
assert_check(
    "-14.79" not in tex,
    "teacher_full '-14.79' NOT present in main.tex"
)
# A3: student_full -9.89 present
assert_check(
    student_r2_s in tex,
    f"student_full 2dp '{student_r2_s}' present in main.tex (raw mean={student_mean:.6f})"
)
# A4: Affine_K-only -7.27 present
assert_check(
    affine_k_r2_s in tex,
    f"Affine_K-only 2dp '{affine_k_r2_s}' present in main.tex (raw mean={affine_k_mean:.6f})"
)
# A5: K-only DeltaLL +7.50 present
assert_check(
    dell_r2_s in tex,
    f"K-only DeltaLL '{dell_r2_s}' present in main.tex ({affine_k_r2_s} - ({teacher_r2_s}))"
)
# A6: K-only DeltaLL +7.52 NOT present
assert_check(
    "+7.52" not in tex,
    "K-only DeltaLL '+7.52' NOT present in main.tex"
)

# ===========================================================================
# Part B: CCA Table 3 verification from w4_cca_perhead.json
# ===========================================================================
print("\n=== Part B: CCA Table 3 ===")

# Pair ordering as in JSON file
pair_names_json = list(cca["results"].keys())

# LaTeX pair labels (JSON key → paper format)
def json_pair_to_latex(pair: str) -> str:
    a, b = pair.split("_")
    return f"{a}$\\to${b}"

# Expected values (cross-check labels)
# We compute from JSON; these are the reference from the plan
expected_pairs = [
    ("8B_0.6B",   0.9944, 0.9899, 75.9),
    ("4B_1.7B",   0.9943, 0.9890, 79.0),
    ("4B_0.6B",   0.9944, 0.9906, 74.6),
    ("8B_1.7B",   0.9944, 0.9884, 79.9),
    ("1.7B_0.6B", 0.9967, 0.9937, 79.9),
    ("8B_4B",     0.9974, 0.9940, 82.3),
]

def compute_pair_stats(pair_data: dict):
    """Compute per-head mean rhoK, rhoV across ALL layers, and pct from JSON."""
    layers = pair_data["layers"]
    all_rhoK = []
    all_rhoV = []
    for layer in layers:
        all_rhoK.extend(layer["rho1_K_per_head"])
        all_rhoV.extend(layer["rho1_V_per_head"])
    rhoK = round(sum(all_rhoK) / len(all_rhoK), 4)
    rhoV = round(sum(all_rhoV) / len(all_rhoV), 4)
    frac = pair_data["fraction_heads_K_gt_V"]
    pct = round(frac * 100, 1)
    return rhoK, rhoV, pct


# Check table header
assert_check(
    r"% heads $\rho_K>\rho_V$" in tex,
    "Table 3 header contains '% heads $\\rho_K>\\rho_V$'"
)
assert_check(
    r"K-only $\dll$" not in tex.split(r"\label{tab:cca}")[1].split(r"\end{table}")[0]
    if r"\label{tab:cca}" in tex else False,
    "Table 3 does NOT contain 'K-only $\\dll$'"
)

# Check table body: extract the tabular block for tab:cca
cca_table_match = re.search(
    r"\\label\{tab:cca\}.*?\\begin\{tabular\}(.*?)\\end\{tabular\}",
    tex, re.DOTALL
)
assert_check(
    cca_table_match is not None,
    "Table 3 (tab:cca) tabular environment found in main.tex"
)

if cca_table_match:
    table_body = cca_table_match.group(1)

    # Verify all 6 data rows
    for idx, (pair_name, exp_rhoK, exp_rhoV, exp_pct) in enumerate(expected_pairs):
        if pair_name not in cca["results"]:
            assert_check(False, f"Pair '{pair_name}' not found in CCA JSON results")
            continue

        rhoK, rhoV, pct = compute_pair_stats(cca["results"][pair_name])

        # Match the values computed from JSON against expected cross-check values
        assert_check(
            rhoK == exp_rhoK,
            f"CCA {pair_name}: rhoK computed={rhoK}, expected={exp_rhoK}"
        )
        assert_check(
            rhoV == exp_rhoV,
            f"CCA {pair_name}: rhoV computed={rhoV}, expected={exp_rhoV}"
        )
        assert_check(
            pct == exp_pct,
            f"CCA {pair_name}: pct computed={pct}, expected={exp_pct}"
        )

        # Check row exists in table
        latex_pair = json_pair_to_latex(pair_name)
        rhoK_s = f"{rhoK:.4f}"
        rhoV_s = f"{rhoV:.4f}"
        pct_s = f"{pct:.1f}"

        # Build expected row pattern (flexible whitespace)
        row_pattern = re.compile(
            re.escape(latex_pair) + r"\s*&\s*"
            + re.escape(rhoK_s) + r"\s*&\s*"
            + re.escape(rhoV_s) + r"\s*&\s*"
            + re.escape(pct_s)
        )
        assert_check(
            row_pattern.search(table_body) is not None,
            f"Table 3 row: {latex_pair} {rhoK_s} {rhoV_s} {pct_s} found"
        )

# Check "across all six pairs" (figure caption)
assert_check(
    "across all six pairs" in tex,
    "Figure caption contains 'across all six pairs'"
)

# Check "74.6--82.3\\%" (dissociation text)
assert_check(
    "74.6--82.3\\%" in tex,
    "Dissociation text contains '74.6--82.3\\%'"
)

# ===========================================================================
# Part C: Forbidden words scan
# ===========================================================================
print("\n=== Part C: Forbidden words scan ===")

forbidden = [
    "1000 resamples",
    "by construction",
    "-14.79",
    "+7.52",
    "monolithic",
    "necessary but not sufficient",
    "75--79",
    "selective joint",
    "+2.61",
    "+2.46",
    "pooled",
]

for word in forbidden:
    assert_check(
        word not in tex,
        f"Forbidden word '{word}' NOT in main.tex"
    )

# ===========================================================================
# Part D: Summary
# ===========================================================================
print("\n=== Summary ===")
n_pass = sum(1 for p, _ in results if p)
n_fail = sum(1 for p, _ in results if not p)

if n_fail == 0:
    print("ALL PASS")
    sys.exit(0)
else:
    print(f"{n_fail} FAILED")
    sys.exit(1)
