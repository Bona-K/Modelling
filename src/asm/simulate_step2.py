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
    theta_same: float = 0.6
    p_edge: float = 0.2
    n_agents: int = 100
    p_stay_anchor: float = 0.3
    p_move_below: float = 0.4     # when threshold fails, prob. to attempt a move (backup rule)
    anti_backtrack: bool = True   # penalize immediate bounce-back to previous node
    seed: int = 1
    count_t0: bool = True

def build_mall(params: Params):
    rng = np.random.default_rng(params.seed)
    G = nx.erdos_renyi_graph(params.n_nodes, params.p_edge, seed=params.seed)
    anchors = {0}                 # (CHANGED) fix the anchor at node 0 for consistent diffusion baseline
    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"  #anchor tagged as 'similar' for Step2 logic
            G.nodes[n]["A"] = 3
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = rng.choice(["similar", "different"])  #keep tenants random for variety
            G.nodes[n]["A"] = 1
    return G, anchors

def is_complement(a: str, b: str) -> bool:
    return a != b                  #complementary definition: 'different' counts as complement

def simulate(params: Params):
    rng = np.random.default_rng(params.seed)
    G, anchors = build_mall(params)
    A0 = next(iter(anchors))
    customers = np.full(params.n_agents, A0, dtype=int)
    prev_pos   = np.full(params.n_agents, A0, dtype=int)   #store previous position to limit ping-pong moves
    footfall = np.zeros(params.n_nodes, dtype=int)

    if params.count_t0:
        for c in customers:
            footfall[c] += 1       

    for _ in range(params.steps):
        for i in range(params.n_agents):
            here = customers[i]
            neighbors = list(G.neighbors(here))
            if not neighbors:
                neighbors = [here]  

            curr_cat = G.nodes[here]["category"]
            same_neighbors = [v for v in neighbors if G.nodes[v]["category"] == curr_cat]
            ratio_same = len(same_neighbors) / len(neighbors) if neighbors else 0.0

            if G.nodes[here]["role"] == "anchor" and rng.random() < params.p_stay_anchor:
                nxt = here

            elif len(same_neighbors) > 0 and ratio_same >= params.theta_same:
                nxt = int(rng.choice(same_neighbors))

            else:
                #backup rule: below threshold, try complementary neighbors first
                comp_neighbors = [v for v in neighbors if is_complement(curr_cat, G.nodes[v]["category"])]

                if rng.random() < params.p_move_below:
                    pool = comp_neighbors if comp_neighbors else neighbors  #prefer complement; fallback to any neighbor
                    nxt = int(rng.choice(pool))
                else:
                    nxt = here  #explicit stay when backup move not taken

            #anti-backtrack: if chosen next node equals previous node, 50% chance to stay instead
            if params.anti_backtrack and nxt == prev_pos[i] and rng.random() < 0.5:
                nxt = here

            prev_pos[i] = here
            customers[i] = nxt
            footfall[nxt] += 1

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall
    })

    ensure_dir("data/outputs")
    df.to_csv("data/outputs/step2.csv", index=False)
    print("[Saved] data/outputs/step2.csv")
    return df

print(simulate(Params()))