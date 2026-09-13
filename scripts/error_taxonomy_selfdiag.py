"""A7 (optional, audit3 section 13): taxonomy of the Self-baseline failures.

The paper reports that Self EM is non-monotonic in student size (0.6B 0.911,
1.7B 0.429, 4B 0.804) and that raw and normalized exact match agree, so the dip
is not a formatting artifact. audit3 asks for one more cheap check: classify the
generations so a reader can see that the failures are ordinary task errors
rather than a pipeline artefact.

Everything here is computed from the archived generations in
`reports/phaseB_selfdiag_seed0.json` (written by `experiments/phaseB_selfdiag.py` with the
`return_gen` change from REVISION_PLAN2 I-4), so no GPU and no model access is
needed. The classifier is deterministic and rule-based:

  no_content          empty or punctuation-only generation
  refusal_or_hedge    denial phrases ("cannot", "not mentioned", ...)
  repetition_loop     a 2-gram repeat covers at least half the tokens
  hit_length_limit    generation reached the 16-token cap with content
  wrong_entity        quotes a synthetic-domain entity or number that is not gold
  other_text          remaining free text

Usage:
  python scripts/error_taxonomy_selfdiag.py [--report reports/phaseB_selfdiag_seed0.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import string

PUNCT = set(string.punctuation)
HEDGES = (
    "dont know", "do not know", "cannot", "can not", "cant ", "unable",
    "no information", "not mentioned", "unknown", "no answer", "insufficient",
    "not specified", "not provided",
)
ENTITY = re.compile(r"\b(proj|comp|plant|tester)-[a-z0-9]+\b")
NUMBER = re.compile(r"\b\d+(\.\d+)?\b")
MAX_NEW = 16


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in PUNCT else " " for ch in s)
    return " ".join(s.split())


def classify(gen: str, answer: str, gen_tokens: int) -> str:
    norm = normalize(gen)
    norm_answer = normalize(answer)
    if norm_answer and norm_answer in norm:
        return "correct"
    if not norm or not any(ch.isalnum() for ch in gen):
        return "no_content"
    if any(h in norm for h in HEDGES):
        return "refusal_or_hedge"
    toks = norm.split()
    if len(toks) >= 4:
        bigrams = [tuple(toks[i:i + 2]) for i in range(len(toks) - 1)]
        counts = collections.Counter(bigrams)
        if counts and counts.most_common(1)[0][1] / len(bigrams) >= 0.5:
            return "repetition_loop"
    if gen_tokens >= MAX_NEW:
        return "hit_length_limit"
    if ENTITY.search(norm) or NUMBER.search(norm):
        return "wrong_entity"
    return "other_text"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/phaseB_selfdiag_seed0.json")
    ap.add_argument("--output", default="reports/error_taxonomy_selfdiag.json")
    a = ap.parse_args()

    with open(a.report, encoding="utf-8") as f:
        report = json.load(f)

    out = {"task": "error_taxonomy_selfdiag", "source": a.report, "students": {}}
    for entry in report["students"]:
        student = entry["student"]
        counts: collections.Counter = collections.Counter()
        failures = []
        for row in entry["rows"]:
            label = classify(row["gen"], row["answer"], int(row.get("gen_tokens", 0)))
            counts[label] += 1
            if label != "correct":
                failures.append({"id": row["id"], "answer": row["answer"],
                                 "gen": row["gen"], "label": label,
                                 "norm_em": row.get("norm_em")})
        n = len(entry["rows"])
        out["students"][student] = {
            "n": n,
            "counts": dict(counts),
            "self_em": entry.get("norm_em"),
            "n_failures": len(failures),
            "raw_fail_norm_pass": entry.get("raw_fail_norm_pass"),
            "failures": failures,
        }
        dist = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"{student:5s} n={n:3d} norm_em={entry.get('norm_em')}  {dist}")
        if student == "1.7B":
            print("     1.7B failure examples:")
            for f in failures[:6]:
                print(f"       [{f['label']:16s}] gold={f['answer'][:22]!r:26s} "
                      f"gen={f['gen'][:40]!r}")

    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[taxonomy] saved: {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
