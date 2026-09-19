"""V3 W7：第二域数据准备（SQuAD-with-context + NQ-Open novelty probe）

背景：预注册为 NQ-with-context，但 nq_open 数据集无 context 字段、Wikipedia API
被网络阻断，无法构造 NQ context。替代方案（决策记录于 ITERATION_LOG）：
- 4-way ablation 数据：SQuAD validation（带 context，答案在文档内，事实型短答案）
  —— 与 NQ-with-context 结构等价（问题 + 维基式上下文文档 + 1-5 token 答案）
- novelty probe 数据：nq_open（开放域事实问答，无 context，测学生先验知识）

输出：
- data/squad_test_seed0.json: 30 样本 {doc, q, answer, hop, domain}
- data/nq_open_probe_seed0.json: 30 样本 {q, answer}（novelty probe 用）

预注册（不事后调整）：seed=0 固定采样 30 个 1-5 token 答案样本。
"""
import argparse, json, os, random
import tempfile
import numpy as np
import requests

_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOT, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")
SQUAD_PARQUET = os.environ.get("V3_SQUAD_PARQUET",
                               os.path.join(tempfile.gettempdir(), "squad_val.parquet"))


def fetch_squad(parquet: str):
    if not os.path.exists(parquet):
        url = "https://hf-mirror.com/datasets/rajpurkar/squad/resolve/main/plain_text/validation-00000-of-00001.parquet"
        r = requests.get(url, timeout=60)
        with open(parquet, "wb") as f:
            f.write(r.content)
        print(f"downloaded squad parquet: {len(r.content)} bytes")
    import pandas as pd
    return pd.read_parquet(parquet)


def build_squad_subset(df, n=30, seed=0, doc_tokens=300):
    rng = random.Random(seed)
    # 过滤：1-5 token 答案
    cand = []
    for _, row in df.iterrows():
        texts = list(row["answers"]["text"]) if isinstance(row["answers"], dict) else list(row["answers"])
        if not texts:
            continue
        t = texts[0]
        if 1 <= len(t.split()) <= 5:
            cand.append(row)
    print(f"squad candidates (1-5 token answers): {len(cand)}")
    picked = rng.sample(cand, min(n, len(cand)))

    out = []
    for i, row in enumerate(picked):
        texts = list(row["answers"]["text"]) if isinstance(row["answers"], dict) else list(row["answers"])
        toks = row["context"].split()
        if len(toks) > doc_tokens:
            toks = toks[:doc_tokens]
        out.append({
            "id": f"squad_{seed}_{i}",
            "doc": " ".join(toks),
            "q": row["question"],
            "answer": texts[0],
            "answer_alt": texts,
            "hop": 1,
            "domain": "squad",
            "title": row["title"],
        })
    return out


def fetch_nq_open():
    from modelscope import MsDataset
    ds = MsDataset.load("google-research-datasets/nq_open", split="validation")
    out = []
    for ex in ds:
        ans = ex.get("answer", [])
        if isinstance(ans, str):
            ans = [ans]
        out.append({"q": ex.get("question", ""), "answer": list(ans)})
    return out


def build_nq_probe(samples, n=30, seed=0):
    rng = random.Random(seed)
    cand = [s for s in samples if s["answer"] and all(1 <= len(a.split()) <= 5 for a in s["answer"])]
    picked = rng.sample(cand, min(n, len(cand)))
    out = []
    for i, s in enumerate(picked):
        out.append({"id": f"nq_{seed}_{i}", "q": s["q"], "answer": s["answer"][0],
                    "answer_alt": s["answer"], "hop": 1, "domain": "nq_open"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doc-tokens", type=int, default=300)
    a = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # SQuAD 4-way ablation 数据
    df = fetch_squad(SQUAD_PARQUET)
    subset = build_squad_subset(df, n=a.n, seed=a.seed, doc_tokens=a.doc_tokens)
    squad_out = f"{DATA_DIR}/squad_test_seed{a.seed}.json"
    with open(squad_out, "w") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    docs = [len(s["doc"].split()) for s in subset]
    print(f"saved {len(subset)} squad samples -> {squad_out}")
    print(f"  doc tokens: mean={np.mean(docs):.0f} min={min(docs)} max={max(docs)}")

    # NQ-Open novelty probe 数据
    nq = fetch_nq_open()
    probe = build_nq_probe(nq, n=a.n, seed=a.seed)
    nq_out = f"{DATA_DIR}/nq_open_probe_seed{a.seed}.json"
    with open(nq_out, "w") as f:
        json.dump(probe, f, ensure_ascii=False, indent=2)
    print(f"saved {len(probe)} nq_open probe samples -> {nq_out}")

    for s in subset[:3]:
        print(f"  Q: {s['q'][:90]}")
        print(f"  A: {s['answer']}  | doc[:80]: {s['doc'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
