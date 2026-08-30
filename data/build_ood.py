"""合成 OOD 域数据引擎（V3 Phase 0 用）。

设计目标（对应 V3 方案 §2 内部效度修复）：
1. 虚构企业域（军工风味的虚构实体），确保 8B/0.6B 预训练权重都不含这些知识；
2. 隐藏知识图谱 K → 程序化渲染文档 D，答案跨句/跨文档才可解（hop>=2）；
3. 问题从 K 采样生成，答案由规则可验证（数值/实体比对）；
4. 按实体簇划分 train/val/test，杜绝泄漏。

虚构词表：所有实体名、编号、数值均为程序生成，与真实世界无对应。
"""
from __future__ import annotations

import hashlib
import json
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


def main():
    ds = build_dataset(n_projects=10, seed=42)
    out_dir = "/workspace/v3/data"
    import os

    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = f"{out_dir}/{split}.json"
        with open(path, "w") as f:
            json.dump(ds[split], f, ensure_ascii=False, indent=1)
        print(f"[{split}] {len(ds[split])} samples -> {path}")
    with open(f"{out_dir}/graph.json", "w") as f:
        json.dump(ds["graph"], f, ensure_ascii=False, indent=1)
    print("graph saved")

    # 抽查
    s = ds["train"][0]
    print("\nSAMPLE doc:", s["doc"][:220], "...")
    print("SAMPLE q:", s["q"], "| ans:", s["answer"], "| hop:", s["hop"])


if __name__ == "__main__":
    main()
