# Sales & Anchor Quality Model Comparison
import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from utils.io import ensure_dir

@dataclass
class Params:
    n_nodes: int = 20
    n_anchors: int = 1
    steps: int = 10
    theta_same: float = 0.5
    p_edge: float = 0.2
    n_agents: int = 100
    p_stay_anchor: float = 0.4
    p_move_below: float = 0.6
    anti_backtrack: bool = True
    seed: int = 1
    count_t0: bool = True

    # --- Step3 additions ---
    comp_bias: float = 1.2                 # mild bias toward complementary
    purchase_prob: dict | None = None      # purchase probability by category
    cool_down: int = 1                     # cooldown after purchase
    anchor_quality: str = "medium"         # for preset tagging

def build_mall(params: Params):
    rng = np.random.default_rng(params.seed)
    G = nx.erdos_renyi_graph(params.n_nodes, params.p_edge, seed=params.seed)
    anchors = {0}  # fixed anchor
    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"
            G.nodes[n]["A"] = 3
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = rng.choice(["similar", "complementary", "different"])
            G.nodes[n]["A"] = 1
    return G, anchors


def is_complement(a: str, b: str) -> bool:
    return a != b and ("complementary" in [a, b])

def simulate(params: Params):
    rng = np.random.default_rng(params.seed)
    G, anchors = build_mall(params)
    A0 = next(iter(anchors))

    customers = np.full(params.n_agents, A0, dtype=int)
    prev_pos = np.full(params.n_agents, A0, dtype=int)
    cool = np.zeros(params.n_agents, dtype=int)  # cooldown tracker
    footfall = np.zeros(params.n_nodes, dtype=int)
    sales = np.zeros(params.n_nodes, dtype=int)
    traces = []

    # Default purchase probs if none provided
    buy_p = params.purchase_prob or {
        "anchor": 0.20,
        "similar": 0.18,
        "complementary": 0.35,
        "different": 0.12,
    }

    if params.count_t0:
        for c in customers:
            footfall[c] += 1
        traces.append(customers.copy())

    for _ in range(params.steps):
        for i in range(params.n_agents):
            if cool[i] > 0:
                cool[i] -= 1
                continue  # skip move if cooling down

            here = customers[i]
            neighbors = list(G.neighbors(here))
            if not neighbors:
                neighbors = [here]

            curr_cat = G.nodes[here]["category"]
            same_neighbors = [v for v in neighbors if G.nodes[v]["category"] == curr_cat]
            ratio_same = len(same_neighbors) / len(neighbors) if neighbors else 0.0

            # anchor quality
            stay_boost = 0.1 if params.anchor_quality == "medium" else 0.25
            p_stay_anchor = params.p_stay_anchor + stay_boost

            if G.nodes[here]["role"] == "anchor" and rng.random() < p_stay_anchor:
                nxt = here

            elif len(same_neighbors) > 0 and ratio_same >= params.theta_same:
                nxt = int(rng.choice(same_neighbors))

            else:
                comp_neighbors = [v for v in neighbors if is_complement(curr_cat, G.nodes[v]["category"])]
                if rng.random() < params.p_move_below:
                    # weighted move
                    weights = np.array([
                        params.comp_bias if G.nodes[v]["category"] == "complementary" else 1.0
                        for v in neighbors
                    ], dtype=float)
                    probs = weights / weights.sum()
                    nxt = int(rng.choice(neighbors, p=probs))
                else:
                    nxt = here

            if params.anti_backtrack and nxt == prev_pos[i] and rng.random() < 0.5:
                nxt = here

            prev_pos[i] = here
            customers[i] = nxt
            footfall[nxt] += 1

            # purchase logic - cooldown after purchase
            cat = G.nodes[nxt]["category"]
            role = G.nodes[nxt]["role"]
            p_buy = buy_p["anchor"] if role == "anchor" else buy_p.get(cat, 0.1)
            if rng.random() < p_buy:
                sales[nxt] += 1
                cool[i] = params.cool_down

        traces.append(customers.copy())

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall,
        "sales": sales,
    })

    ensure_dir("data/outputs")
    df.to_csv("data/outputs/step3.csv", index=False)
    print("[Saved] data/outputs/step3.csv")

    # record exits
    exit_df = pd.DataFrame(columns=["agent", "tick", "node"])

    class SimResult: pass
    sim = SimResult()
    sim.mall_graph = G
    sim.traces = traces
    sim.p = params

    return df, exit_df, A0

# Model comparison experiment presets
def preset_A() -> Params:
    """Medium anchor, mild comp bias, strong complementary purchase."""
    return Params(
        anchor_quality="medium",
        p_stay_anchor=0.35,
        p_move_below=0.6,
        comp_bias=1.2,
        purchase_prob={"anchor":0.20, "similar":0.18, "complementary":0.35, "different":0.12},
        cool_down=1,
    )


def preset_B() -> Params:
    """High anchor, strong stickiness, slightly lower comp bias."""
    return Params(
        anchor_quality="high",
        p_stay_anchor=0.65,
        p_move_below=0.5,
        comp_bias=1.1,
        purchase_prob={"anchor":0.22, "similar":0.16, "complementary":0.33, "different":0.10},
        cool_down=2,
    )


if __name__ == "__main__":
    df, exit_df, A0 = simulate(preset_A())
    print("Anchor fixed at node:", A0)
    print(df.sort_values("sales", ascending=False).head())