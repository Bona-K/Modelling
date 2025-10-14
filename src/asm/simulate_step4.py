# Adapt Boid Rule
import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from collections import Counter
from typing import Tuple, Dict, List, Optional

@dataclass
class Params:
    n_nodes: int = 20
    p_edge: float = 0.2
    n_agents: int = 100
    steps: int = 10
    seed: int = 1
    count_t0: bool = True

    p_stay_anchor: float = 0.25
    theta_same: float = 0.5
    p_move_below: float = 0.6
    anti_backtrack: bool = True
    comp_bias: float = 1.2  

    purchase_prob: Dict[str, float] = None

    # Boids on graph
    k_hop: int = 1                 # local field radius
    w_center: float = 0.8          # cohesion weight
    w_avoid: float = 1.2           # separation weight (penalty is negative inside)
    w_align: float = 0.6           # alignment weight
    w_love: float = 1.0            # target("carrot") weight
    max_node_capacity: int = 15    # soft capacity for separation scaling

    # anchor quality
    anchor_quality: int = 3

    def __post_init__(self):
        if self.purchase_prob is None:
            self.purchase_prob = {
                "anchor": 0.20,
                "similar": 0.18,
                "complementary": 0.35,
                "different": 0.12,
            }

def build_mall(p: Params):
    rng = np.random.default_rng(p.seed)
    G = nx.erdos_renyi_graph(p.n_nodes, p.p_edge, seed=p.seed)
    anchors = {0}  # fix anchor at 0
    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"  
            G.nodes[n]["A"] = p.anchor_quality
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = rng.choice(["similar", "complementary", "different"])
            G.nodes[n]["A"] = 1
    return G, anchors

def is_complement(a: str, b: str) -> bool:
    return (a != b) and (b == "complementary")

def shortest_next_step(G: nx.Graph, here: int, target: int) -> Optional[int]:
    if here == target:
        return None
    try:
        path = nx.shortest_path(G, here, target)
        if len(path) >= 2:
            return path[1]
        return None
    except nx.NetworkXNoPath:
        return None

def pick_complement_target(G: nx.Graph, here: int) -> Optional[int]:
    comps = [n for n in G.nodes if G.nodes[n].get("category") == "complementary"]
    if not comps:
        return None
    dists = []
    for n in comps:
        try:
            d = nx.shortest_path_length(G, here, n)
            dists.append((d, n))
        except nx.NetworkXNoPath:
            continue
    if not dists:
        return None
    dmin = min(dists)[0]
    candidates = [n for d, n in dists if d == dmin]
    return int(np.random.choice(candidates))

def k_hop_ball(G: nx.Graph, center: int, k: int) -> List[int]:
    # nodes within k hops (include center)
    return list(nx.single_source_shortest_path_length(G, center, cutoff=k).keys())

def local_counts(positions: np.ndarray, k_nodes: List[int]) -> Counter:
    c = Counter()
    for a in positions:
        aa = int(a)
        if aa in k_nodes:
            c[aa] += 1
    return c

def simulate(p: Params):
    rng = np.random.default_rng(p.seed)
    G, anchors = build_mall(p)
    A0 = next(iter(anchors))

    customers = np.full(p.n_agents, A0, dtype=int)
    prev_pos = np.full(p.n_agents, A0, dtype=int)
    footfall = np.zeros(p.n_nodes, dtype=int)
    sales = np.zeros(p.n_nodes, dtype=int)

    last_dest_hist = Counter()  # for alignment
    traces: List[np.ndarray] = []

    # diagnostics accumulators: sum of boid components for chosen arrivals
    diag_sum = {
        "coh": np.zeros(p.n_nodes, dtype=float),
        "sep": np.zeros(p.n_nodes, dtype=float),   # store as positive pressure
        "ali": np.zeros(p.n_nodes, dtype=float),
        "love": np.zeros(p.n_nodes, dtype=float),
        "arrivals": np.zeros(p.n_nodes, dtype=int),
    }

    if p.count_t0:
        for c in customers:
            footfall[c] += 1
        traces.append(customers.copy())

    for _t in range(p.steps):
        snapshot = customers.copy()

        for i in range(p.n_agents):
            here = int(customers[i])
            neighbors = list(G.neighbors(here)) or [here]

            curr_cat = G.nodes[here]["category"]
            same_neighbors = [v for v in neighbors if G.nodes[v]["category"] == curr_cat]
            ratio_same = len(same_neighbors) / len(neighbors) if neighbors else 0.0
            base_scores = np.ones(len(neighbors), dtype=float)

            if G.nodes[here]["role"] == "anchor" and rng.random() < p.p_stay_anchor:
                nxt = here
            elif same_neighbors and ratio_same >= p.theta_same:
                nxt = int(rng.choice(same_neighbors))
            else:
                if rng.random() < p.p_move_below:
                    weights = np.array([
                        p.comp_bias if is_complement(curr_cat, G.nodes[v]["category"]) else 1.0
                        for v in neighbors
                    ], dtype=float)
                    probs = weights / weights.sum()
                    nxt = int(rng.choice(neighbors, p=probs))
                else:
                    nxt = here

            # boids scores over neighbor options
            k_nodes = k_hop_ball(G, here, p.k_hop)
            occ = local_counts(snapshot, k_nodes)

            coh = np.array([occ.get(v, 0) for v in neighbors], dtype=float)
            cap = max(1, p.max_node_capacity)
            sep = -np.array([occ.get(v, 0) / cap for v in neighbors], dtype=float)  # penalty -> negative
            ali = np.array([last_dest_hist.get(v, 0) for v in neighbors], dtype=float)

            love = np.zeros(len(neighbors), dtype=float)
            carrot = pick_complement_target(G, here)
            if carrot is not None:
                step_to = shortest_next_step(G, here, carrot)
                if step_to is not None:
                    for idx, v in enumerate(neighbors):
                        if v == step_to:
                            love[idx] = 1.0

            boid_scores = (
                p.w_center * coh +
                p.w_avoid * sep +
                p.w_align * ali +
                p.w_love * love
            )

            cand_nodes = neighbors + ([here] if here not in neighbors else [])
            if here not in neighbors:
                base_scores = np.append(base_scores, 1.0)
                coh = np.append(coh, occ.get(here, 0))
                sep = np.append(sep, -occ.get(here, 0) / cap)
                ali = np.append(ali, last_dest_hist.get(here, 0))
                love = np.append(love, 0.0)
                boid_scores = np.append(boid_scores, 0.0)

            total_scores = base_scores + boid_scores
            shift = max(0.0, -total_scores.min()) + 1e-6
            probs = (total_scores + shift) / (total_scores + shift).sum()

            # sample next and remember the chosen boid components
            nxt_idx = int(rng.choice(np.arange(len(cand_nodes)), p=probs))
            nxt = int(cand_nodes[nxt_idx])

            # collect diagnostics for chosen arrival
            diag_sum["coh"][nxt]  += coh[nxt_idx]
            diag_sum["sep"][nxt]  += -sep[nxt_idx]  
            diag_sum["ali"][nxt]  += ali[nxt_idx]
            diag_sum["love"][nxt] += love[nxt_idx]
            diag_sum["arrivals"][nxt] += 1

            # anti-backtrack
            if p.anti_backtrack and nxt == prev_pos[i] and rng.random() < 0.5:
                nxt = here

            prev_pos[i] = here
            customers[i] = nxt
            footfall[nxt] += 1

            # purchase attempt
            role = G.nodes[nxt]["role"]
            key = "anchor" if role == "anchor" else G.nodes[nxt]["category"]
            if rng.random() < p.purchase_prob.get(key, 0.0):
                sales[nxt] += 1

        last_dest_hist = Counter(int(x) for x in customers)
        traces.append(customers.copy())

    # outputs
    df = pd.DataFrame({
        "node": np.arange(p.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall,
        "sales": sales,
    })

    # build per-node diagnostics (averaged per arrival)
    arr = np.maximum(1, diag_sum["arrivals"])
    diag_df = pd.DataFrame({
        "node": np.arange(p.n_nodes),
        "coh_mean":  diag_sum["coh"]  / arr,
        "sep_mean":  diag_sum["sep"]  / arr,
        "ali_mean":  diag_sum["ali"]  / arr,
        "love_mean": diag_sum["love"] / arr,
        "arrivals":  diag_sum["arrivals"],
    })

    exit_df = pd.DataFrame({"tick": [], "exited": []})

    class SimResult: pass
    sim = SimResult()
    sim.mall_graph = G
    sim.traces = traces
    sim.p = p

    # return diag_df as 4th element
    return df, exit_df, A0, diag_df

# Presets (A/B)
def preset_A() -> Params:
    # more exploratory
    return Params(
        steps=10, n_agents=100,
        p_stay_anchor=0.20,
        comp_bias=1.2,
        w_center=0.8, w_avoid=1.2, w_align=0.6, w_love=1.0,
        k_hop=1, max_node_capacity=15,
        anchor_quality=3,
    )

def preset_B() -> Params:
    # more funneling
    return Params(
        steps=10, n_agents=100,
        p_stay_anchor=0.45,
        comp_bias=1.1,
        w_center=0.6, w_avoid=1.0, w_align=0.6, w_love=1.8,
        k_hop=1, max_node_capacity=12,
        anchor_quality=4,
    )

# Quick stats (for notebook)
def quick_stats(df: pd.DataFrame) -> Dict[str, float]:
    cov = (df["footfall"] > 0).mean()
    top3 = df.sort_values("sales", ascending=False)["sales"].head(3).sum()
    total_sales = max(1, int(df["sales"].sum()))
    comp_sales = df.loc[df["category"] == "complementary", "sales"].sum()
    return {
        "Coverage": cov,
        "Top-3 Sales Share": top3 / total_sales,
        "Complementary Sales Share": comp_sales / total_sales,
    }
