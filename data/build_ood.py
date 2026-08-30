"""合成 OOD 域数据引擎（V3 Phase 0 用）。

设计目标（对应 V3 方案 §2 内部效度修复）：
1. 虚构企业域（军工风味的虚构实体），确保 8B/0.6B 预训练权重都不含这些知识；
2. 隐藏知识图谱 K → 程序化渲染文档 D，答案跨句/跨文档才可解（hop>=2）；
3. 问题从 K 采样生成，答案由规则可验证（数值/实体比对）；
4. 按实体簇划分 train/val/test，杜绝泄漏。

虚构词表：所有实体名、编号、数值均为程序生成，与真实世界无对应。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 虚构词表（全虚构，防止权重记忆）
# ---------------------------------------------------------------------------

# 虚构机构/地点/部件/人员词根（Aurelia Aerospace Group）
ORG = ["Aurelia", "Vesper", "Corvus", "Meridian", "Halcyon", "Nimbus", "Cinder", "Obsidian", "Zephyr", "Ironpeak"]
PLANT = ["Plant-{}".format(c) for c in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]]
COMPONENT = ["Comp-{}".format(c) for c in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1"]]
MATERIAL = ["Mat-{}".format(c) for c in ["X1", "X2", "X3", "Y1", "Y2", "Y3", "Z1", "Z2", "Z3", "W1"]]
TESTER = ["Tester-{}".format(c) for c in ["Q1", "Q2", "Q3", "R1", "R2", "R3", "S1", "S2", "S3", "T1"]]
PROJECT = ["Proj-{}".format(c) for c in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]]

# ---------------------------------------------------------------------------
# 隐藏知识图谱
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """虚构实体：项目/部件/工厂/材料/测试员。"""

    name: str
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Graph:
    """隐藏知识图谱：实体集合 + 关系集合。

    关系：project→component, component→material, component→plant,
          plant→tester, project→{attr...}
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, rel, tgt)

    def add(self, e: Entity):
        self.entities[e.name] = e

    def edge(self, src: str, rel: str, tgt: str):
        self.edges.append((src, rel, tgt))


def build_graph(seed: int = 42) -> Graph:
    """构建一个可复现的虚构知识图谱。"""
    rng = random.Random(seed)
    g = Graph()

    # 项目实体（每个项目有若干数值属性，供数值推理）
    for p in PROJECT:
        g.add(Entity(
            p,
            {
                "budget": str(rng.randint(80, 900)),        # 百万
                "timeline_months": str(rng.randint(6, 48)),
                "weight_tonnes": str(round(rng.uniform(1, 30), 1)),
                "range_km": str(rng.randint(200, 3000)),
            },
        ))

    for c in COMPONENT:
        g.add(Entity(c, {"cost": str(rng.randint(5, 90))}))  # 百万
    for m in MATERIAL:
        g.add(Entity(m, {"density": str(round(rng.uniform(0.5, 5.0), 2))}))
    for pl in PLANT:
        g.add(Entity(pl, {"output_rate": str(rng.randint(100, 900))}))
    for t in TESTER:
        g.add(Entity(t, {"capacity": str(rng.randint(10, 99))}))

    # 边：每个项目绑定 2 个部件；每个部件绑定材料、工厂；工厂绑定测试员
    for p in PROJECT:
        comps = rng.sample(COMPONENT, 2)
        for c in comps:
            g.edge(p, "uses", c)
    for c in COMPONENT:
        g.edge(c, "made_of", rng.choice(MATERIAL))
        g.edge(c, "built_at", rng.choice(PLANT))
    for pl in PLANT:
        g.edge(pl, "certified_by", rng.choice(TESTER))

    return g


def graph_to_json(g: Graph) -> dict:
    return {
        "entities": {n: e.attrs for n, e in g.entities.items()},
        "edges": g.edges,
    }


# ---------------------------------------------------------------------------
# 文档渲染
# ---------------------------------------------------------------------------


def render_doc(g: Graph, project: str, seed: int) -> str:
    """把项目相关的图谱子图渲染成一段自然语言文档。

    故意混入若干无关句子（噪声），使答案需要"跨句定位 + 多跳"。
    """
    rng = random.Random(seed)
    comps = [t for (s, rel, t) in g.edges if s == project and rel == "uses"]
    lines = [f"Project status report for {project}."]
    proj_attrs = g.entities[project].attrs
    lines.append(
        f"The project has a budget of {proj_attrs['budget']} million, "
        f"a target timeline of {proj_attrs['timeline_months']} months, "
        f"weight {proj_attrs['weight_tonnes']} tonnes, and range {proj_attrs['range_km']} km."
    )
    # 噪声句
    noise = [
        "The canteen serves rice and stew on weekdays.",
        "A new parking lot was completed near gate two.",
        "The annual picnic is scheduled for the third quarter.",
        "Maintenance crews rotate on a twelve-hour shift.",
    ]
    lines.append(rng.choice(noise))
    for c in comps:
        mat = [t for (s, rel, t) in g.edges if s == c and rel == "made_of"][0]
        plant = [t for (s, rel, t) in g.edges if s == c and rel == "built_at"][0]
        lines.append(
            f"Component {c} is assembled at {plant}, and its primary material is {mat}."
        )
        lines.append(rng.choice(noise))
    # 工厂详情（保证 hop-2 需要跨句）
    for c in comps:
        plant = [t for (s, rel, t) in g.edges if s == c and rel == "built_at"][0]
        tester = [t for (s, rel, t) in g.edges if s == plant and rel == "certified_by"][0]
        plant_attrs = g.entities[plant].attrs
        lines.append(
            f"{plant} operates at an output rate of {plant_attrs['output_rate']} units per day."
        )
        lines.append(
            f"Quality assurance at {plant} is certified by {tester}, "
            f"which has a testing capacity of {g.entities[tester].attrs['capacity']} units."
        )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 问题生成（hop-1..4，答案规则可验证）
# ---------------------------------------------------------------------------


def generate_questions(g: Graph, project: str, seed: int) -> list[dict]:
    """按图谱生成一组多跳问题，答案来自图谱（可验证）。

    返回 [{q, answer, hop}]
    """
    rng = random.Random(seed)
    proj = g.entities[project]
    comps = [t for (s, rel, t) in g.edges if s == project and rel == "uses"]
    qs = []

    # hop-1：直接属性
    qs.append({
        "q": f"What is the budget (in million) of {project}?",
        "answer": proj.attrs["budget"],
        "hop": 1,
    })
    qs.append({
        "q": f"Which component is used by {project}?",
        "answer": comps[0],
        "hop": 1,
    })
    # hop-2：project→component→plant
    c0 = comps[0]
    plant0 = [t for (s, rel, t) in g.edges if s == c0 and rel == "built_at"][0]
    mat0 = [t for (s, rel, t) in g.edges if s == c0 and rel == "made_of"][0]
    qs.append({
        "q": f"At which plant is the component {c0} used by {project} assembled?",
        "answer": plant0,
        "hop": 2,
    })
    qs.append({
        "q": f"What material is the component {c0} used by {project} made of?",
        "answer": mat0,
        "hop": 2,
    })
    # hop-3：project→component→plant→tester
    tester0 = [t for (s, rel, t) in g.edges if s == plant0 and rel == "certified_by"][0]
    qs.append({
        "q": f"Which tester certifies quality at the plant that assembles {c0}, "
             f"the component used by {project}?",
        "answer": tester0,
        "hop": 3,
    })
    # hop-4：project→component→plant→output_rate（数值链）
    qs.append({
        "q": f"What is the daily output rate of the plant that assembles {c0}, "
             f"which is used by {project}?",
        "answer": g.entities[plant0].attrs["output_rate"],
        "hop": 4,
    })
    # 数值比较（hop-3+）：range 与重量无关，仅测试多属性推理
    qs.append({
        "q": f"Is the range of {project} greater than 1500 km?",
        "answer": "yes" if int(proj.attrs["range_km"]) > 1500 else "no",
        "hop": 1,
    })
    return qs


# ---------------------------------------------------------------------------
# 数据集划分（按实体簇）
# ---------------------------------------------------------------------------


def build_dataset(n_projects: int = 10, seed: int = 42) -> dict[str, list]:
    """构建完整数据集：按项目分 train/val/test。

    n_projects 个项目中，前 6 个为 train，2 个 val，2 个 test。
    """
    g = build_graph(seed)
    projects = PROJECT[:n_projects]
    train_p = projects[:6]
    val_p = projects[6:8]
    test_p = projects[8:]

    def make_split(plist: list[str], split: str) -> list[dict]:
        out = []
        for i, p in enumerate(plist):
            doc = render_doc(g, p, seed + i)
            for q in generate_questions(g, p, seed + 100 + i):
                q["project"] = p
                q["split"] = split
                q["doc"] = doc
                q["id"] = hashlib.md5(f"{split}-{p}-{q['q']}".encode()).hexdigest()[:12]
                out.append(q)
        return out

    return {
        "train": make_split(train_p, "train"),
        "val": make_split(val_p, "val"),
        "test": make_split(test_p, "test"),
        "graph": graph_to_json(g),
    }


# ---------------------------------------------------------------------------
# V2: Expanded dataset with disjoint entity pools & per-seed generation
# ---------------------------------------------------------------------------


def _gen_names(prefix: str, n: int) -> list[str]:
    """Generate n unique entity names: prefix-001, prefix-002, ..."""
    return [f"{prefix}-{i:03d}" for i in range(1, n + 1)]


def build_graph_v2(
    n_train: int = 10,
    n_val: int = 4,
    n_test: int = 8,
    seed: int = 42,
) -> tuple[Graph, dict[str, list[str]], dict[str, set[str]]]:
    """Build knowledge graph with DISJOINT entity pools per split.

    Returns (graph, split_projects, entity_sets) where:
      - split_projects: {"train": [...], "val": [...], "test": [...]}
      - entity_sets: {"train": set(...), "val": set(...), "test": set(...)}
    """
    rng = random.Random(seed)
    g = Graph()
    n_total = n_train + n_val + n_test
    c2 = 2  # components per project

    # --- Project entities ---
    all_projects = _gen_names("Proj-P", n_total)
    train_projects = all_projects[:n_train]
    val_projects = all_projects[n_train:n_train + n_val]
    test_projects = all_projects[n_train + n_val:]

    for p in all_projects:
        g.add(Entity(p, {
            "budget": str(rng.randint(80, 900)),
            "timeline_months": str(rng.randint(6, 48)),
            "weight_tonnes": str(round(rng.uniform(1, 30), 1)),
            "range_km": str(rng.randint(200, 3000)),
        }))

    # --- Disjoint component pools per split ---
    train_comps = _gen_names("Comp-T", n_train * c2)
    val_comps = _gen_names("Comp-V", n_val * c2)
    test_comps = _gen_names("Comp-S", n_test * c2)
    for c in train_comps + val_comps + test_comps:
        g.add(Entity(c, {"cost": str(rng.randint(5, 90))}))

    # --- Disjoint material pools per split ---
    train_mats = _gen_names("Mat-T", n_train * c2)
    val_mats = _gen_names("Mat-V", n_val * c2)
    test_mats = _gen_names("Mat-S", n_test * c2)
    for m in train_mats + val_mats + test_mats:
        g.add(Entity(m, {"density": str(round(rng.uniform(0.5, 5.0), 2))}))

    # --- Disjoint plant pools per split ---
    train_plants = _gen_names("Plant-T", n_train * c2)
    val_plants = _gen_names("Plant-V", n_val * c2)
    test_plants = _gen_names("Plant-S", n_test * c2)
    for pl in train_plants + val_plants + test_plants:
        g.add(Entity(pl, {"output_rate": str(rng.randint(100, 900))}))

    # --- Disjoint tester pools per split ---
    train_testers = _gen_names("Test-T", n_train * c2)
    val_testers = _gen_names("Test-V", n_val * c2)
    test_testers = _gen_names("Test-S", n_test * c2)
    for t in train_testers + val_testers + test_testers:
        g.add(Entity(t, {"capacity": str(rng.randint(10, 99))}))

    # --- Wire edges: project→components→material+plant→tester ---
    def _wire(projects, comps, mats, plants, testers):
        ci = mi = pi = ti = 0
        for p in projects:
            for _ in range(c2):
                c = comps[ci]
                g.edge(p, "uses", c)
                g.edge(c, "made_of", mats[mi])
                g.edge(c, "built_at", plants[pi])
                ci += 1; mi += 1; pi += 1
        for pl in plants:
            g.edge(pl, "certified_by", testers[ti])
            ti += 1

    _wire(train_projects, train_comps, train_mats, train_plants, train_testers)
    _wire(val_projects, val_comps, val_mats, val_plants, val_testers)
    _wire(test_projects, test_comps, test_mats, test_plants, test_testers)

    split_projects = {
        "train": train_projects,
        "val": val_projects,
        "test": test_projects,
    }
    entity_sets = {
        "train": (set(train_projects) | set(train_comps) | set(train_mats)
                  | set(train_plants) | set(train_testers)),
        "val":   (set(val_projects) | set(val_comps) | set(val_mats)
                  | set(val_plants) | set(val_testers)),
        "test":  (set(test_projects) | set(test_comps) | set(test_mats)
                  | set(test_plants) | set(test_testers)),
    }

    return g, split_projects, entity_sets


def build_dataset_v2(
    n_train: int = 10,
    n_val: int = 4,
    n_test: int = 8,
    data_seed: int = 0,
    graph_seed: int = 42,
) -> dict:
    """Build V2 dataset with disjoint entity pools and per-seed support.

    data_seed controls document rendering and question generation randomness.
    graph_seed controls entity attribute values.
    """
    g, split_projects, entity_sets = build_graph_v2(
        n_train, n_val, n_test, seed=graph_seed,
    )

    def make_split(projects: list[str], split_name: str) -> list[dict]:
        out = []
        for i, p in enumerate(projects):
            doc = render_doc(g, p, data_seed + i)
            for q in generate_questions(g, p, data_seed + 100 + i):
                q["project"] = p
                q["split"] = split_name
                q["doc"] = doc
                q["id"] = hashlib.md5(
                    f"{split_name}-{p}-{q['q']}-{data_seed}".encode()
                ).hexdigest()[:12]
                out.append(q)
        return out

    return {
        "train": make_split(split_projects["train"], "train"),
        "val": make_split(split_projects["val"], "val"),
        "test": make_split(split_projects["test"], "test"),
        "graph": graph_to_json(g),
        "_entity_sets": entity_sets,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_entity_overlap(entity_sets: dict[str, set[str]]) -> dict:
    """Check pairwise entity-set intersection between splits."""
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    result: dict = {"pass": True, "details": {}}
    for a, b in pairs:
        overlap = entity_sets[a] & entity_sets[b]
        result["details"][f"{a}-{b}"] = {
            "overlap_count": len(overlap),
            "overlap_entities": sorted(overlap)[:20],
        }
        if overlap:
            result["pass"] = False
    return result


def count_hops(samples: list[dict]) -> dict:
    """Count samples per hop level."""
    hops: dict[int, int] = {}
    for s in samples:
        h = s["hop"]
        hops[h] = hops.get(h, 0) + 1
    return dict(sorted(hops.items()))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="V3 OOD data engine")
    parser.add_argument("--legacy", action="store_true",
                        help="Generate legacy dataset (backward compat)")
    parser.add_argument("--n-train-projects", type=int, default=10)
    parser.add_argument("--n-val-projects", type=int, default=4)
    parser.add_argument("--n-test-projects", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated data seeds for multi-seed generation")
    parser.add_argument("--output-dir", type=str, default="/workspace/v3/data")
    parser.add_argument("--report-dir", type=str, default="/workspace/v3/reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.legacy:
        ds = build_dataset(n_projects=10, seed=42)
        for split in ["train", "val", "test"]:
            path = f"{args.output_dir}/{split}.json"
            with open(path, "w") as f:
                json.dump(ds[split], f, ensure_ascii=False, indent=1)
            print(f"[{split}] {len(ds[split])} samples -> {path}")
        with open(f"{args.output_dir}/graph.json", "w") as f:
            json.dump(ds["graph"], f, ensure_ascii=False, indent=1)
        print("graph saved")
        s = ds["train"][0]
        print("\nSAMPLE doc:", s["doc"][:220], "...")
        print("SAMPLE q:", s["q"], "| ans:", s["answer"], "| hop:", s["hop"])
        return

    # V2 mode
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else [args.data_seed])

    all_verification: dict = {}
    all_hop_counts: dict = {}

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Generating with data_seed={seed}  graph_seed={args.graph_seed}")
        print(f"{'='*50}")

        ds = build_dataset_v2(
            n_train=args.n_train_projects,
            n_val=args.n_val_projects,
            n_test=args.n_test_projects,
            data_seed=seed,
            graph_seed=args.graph_seed,
        )

        # Save per-seed files
        for split in ["train", "val", "test"]:
            path = f"{args.output_dir}/{split}_v2_seed{seed}.json"
            with open(path, "w") as f:
                json.dump(ds[split], f, ensure_ascii=False, indent=1)
            print(f"  [{split}] {len(ds[split])} samples -> {path}")

        # Save graph for this seed
        graph_path = f"{args.output_dir}/graph_v2_seed{seed}.json"
        with open(graph_path, "w") as f:
            json.dump(ds["graph"], f, ensure_ascii=False, indent=1)

        # Verify entity overlap
        entity_sets = ds["_entity_sets"]
        overlap = verify_entity_overlap(entity_sets)
        all_verification[f"seed{seed}"] = {
            "overlap_check": overlap,
            "entity_counts": {k: len(v) for k, v in entity_sets.items()},
        }
        status = "PASS" if overlap["pass"] else "FAIL"
        print(f"  Entity overlap: {status}")
        if not overlap["pass"]:
            for pair, info in overlap["details"].items():
                if info["overlap_count"] > 0:
                    print(f"    {pair}: {info['overlap_count']} overlapping -> {info['overlap_entities']}")

        # Hop decomposition
        hop_info = {}
        for split in ["train", "val", "test"]:
            hop_info[split] = count_hops(ds[split])
        all_hop_counts[f"seed{seed}"] = hop_info
        print(f"  Hop decomposition:")
        for split, hc in hop_info.items():
            print(f"    {split}: {hc}")

    # Save verification report
    os.makedirs(args.report_dir, exist_ok=True)
    report = {
        "n_train_projects": args.n_train_projects,
        "n_val_projects": args.n_val_projects,
        "n_test_projects": args.n_test_projects,
        "samples_per_project": 7,
        "expected_counts": {
            "train": args.n_train_projects * 7,
            "val": args.n_val_projects * 7,
            "test": args.n_test_projects * 7,
        },
        "seeds": seeds,
        "verification": all_verification,
        "hop_decomposition": all_hop_counts,
    }
    report_path = f"{args.report_dir}/w1_data_expansion.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nVerification report -> {report_path}")


if __name__ == "__main__":
    main()
