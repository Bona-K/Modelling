# src/graph.py
"""
Mall graph helpers:
- build_anchor_center_grid: 작은 격자 레이아웃 생성(앵커 중앙)
- save/load as node-link JSON
"""
from typing import Dict, List, Tuple
import random
import json
import networkx as nx

def build_anchor_center_grid(
    side: int = 5,
    categories: List[str] = None,
    seed: int = 42
) -> nx.Graph:
    """
    side x side 격자 그래프 생성. 중앙 노드는 앵커.
    나머지 노드에 카테고리/퀄리티/용량 속성 부여.
    """
    assert side % 2 == 1, "side must be odd (to have a single center)."
    rng = random.Random(seed)
    if categories is None:
        categories = ["ANCHOR","FNB","FASHION","SERVICE","GROCERY","LIFESTYLE"]

    G = nx.grid_2d_graph(side, side)  # nodes: (i,j)
    G = nx.convert_node_labels_to_integers(G, label_attribute="coord")

    center = (side * side) // 2
    for n, data in G.nodes(data=True):
        data["category"] = rng.choice([c for c in categories if c != "ANCHOR"])
        data["is_anchor"] = False
        data["quality"] = round(rng.uniform(0.8, 1.2), 2)
        data["capacity"] = rng.randint(40, 120)

    # 중앙을 앵커로
    G.nodes[center]["category"] = "ANCHOR"
    G.nodes[center]["is_anchor"] = True
    G.nodes[center]["quality"] = 1.3
    G.nodes[center]["capacity"] = 300

    # hop 계산을 위해 anchor id 반환용 속성 저장
    G.graph["anchor_id"] = center
    return G

def node_link_dumps(G: nx.Graph) -> str:
    return json.dumps(nx.readwrite.json_graph.node_link_data(G), ensure_ascii=False, indent=2)

def node_link_dump(G: nx.Graph, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(node_link_dumps(G))

def node_link_load(path: str) -> nx.Graph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.readwrite.json_graph.node_link_graph(data)
