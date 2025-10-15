import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from utils.io import ensure_dir

@dataclass
class Params:
    n_nodes: int = 20
    n_anchors: int = 1
    steps: int = 50
    theta_same: float = 0.6
    p_edge: float = 0.2
    n_agents: int = 200
    p_stay_anchor: float = 0.2
    p_move_below: float = 0.5
    anti_backtrack: bool = True
    backtrack_penalty: float = 0.5
    seed: int = 1
    count_t0: bool = True
    init_spread_ratio: float = 0.30
    agent_marker_size: int = 90
    jitter: float = 0.012
    interval_ms: int = 200

def build_mall(params: Params):
    rng = np.random.default_rng(params.seed)
    G = nx.erdos_renyi_graph(params.n_nodes, params.p_edge, seed=params.seed)
    anchors = {0}

    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"
            G.nodes[n]["A"] = 3
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = rng.choice(["similar", "different"])
            G.nodes[n]["A"] = 1
    return G, anchors

def is_complement(a: str, b: str) -> bool:
    return a != b

def simulate(params: Params):
    rng = np.random.default_rng(params.seed)
    G, anchors = build_mall(params)
    A0 = next(iter(anchors))
    footfall = np.zeros(params.n_nodes, dtype=int)

    customers = np.full(params.n_agents, A0, dtype=int)
    neighbors_A0 = list(G.neighbors(A0)) or [A0]
    k = int(params.init_spread_ratio * params.n_agents)
    if k > 0:
        idx = rng.choice(params.n_agents, size=k, replace=False)
        customers[idx] = rng.choice(neighbors_A0, size=k, replace=True)

    prev_pos = customers.copy()
    traces = []

    if params.count_t0:
        footfall += np.bincount(customers, minlength=params.n_nodes)
        traces.append(customers.copy())

    for _ in range(params.steps):
        for i in range(params.n_agents):
            here = customers[i]
            neighbors = list(G.neighbors(here)) or [here]
            curr_cat = G.nodes[here]["category"]

            same_neighbors = [v for v in neighbors if G.nodes[v]["category"] == curr_cat]
            ratio_same = len(same_neighbors) / len(neighbors)

            
            if G.nodes[here]["role"] == "anchor" and rng.random() < params.p_stay_anchor:
                nxt = here
            
            elif same_neighbors and ratio_same >= params.theta_same:
                nxt = int(rng.choice(same_neighbors))
            else:
               
                comp_neighbors = [v for v in neighbors if is_complement(curr_cat, G.nodes[v]["category"])]
                if rng.random() < params.p_move_below:
                    pool = comp_neighbors if comp_neighbors else neighbors
                    nxt = int(rng.choice(pool))
                else:
                    nxt = here

            
            if params.anti_backtrack and nxt == prev_pos[i] and rng.random() < params.backtrack_penalty:
                nxt = here

            prev_pos[i] = here
            customers[i] = nxt

        footfall += np.bincount(customers, minlength=params.n_nodes)
        traces.append(customers.copy())

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall
    })

    ensure_dir("data/outputs")
    df.to_csv("data/outputs/step2_visible.csv", index=False)
    print("[Saved] data/outputs/step2_visible.csv")

    class SimResult: ...
    sim = SimResult()
    sim.mall_graph = G
    sim.traces = traces
    sim.p = params
    return df, sim, A0

def animate_movement(sim, filename="figs/step2_asm.gif", pos=None, interval=400):
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.animation import FuncAnimation, PillowWriter

    G, traces = sim.mall_graph, sim.traces
    if not traces: raise ValueError("sim.traces is empty")
    if pos is None: pos = nx.spring_layout(G, seed=sim.p.seed, k=0.8)

    cmap = {"anchor":"tab:red","similar":"tab:blue","different":"tab:green"}
    node_colors = [
        (cmap["anchor"] if G.nodes[n].get("role")=="anchor"
         else cmap.get(G.nodes[n].get("category"), "gray"))
        for n in G.nodes
    ]

    fig, ax = plt.subplots(figsize=(7,6)); ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.3, width=1.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=200)

    handles = [
        plt.Line2D([0],[0], marker="o", ls="", color=cmap["anchor"],   label="anchor"),
        plt.Line2D([0],[0], marker="o", ls="", color=cmap["similar"],  label="similar"),
        plt.Line2D([0],[0], marker="o", ls="", color=cmap["different"],label="different"),
        plt.Line2D([0],[0], marker="o", ls="", color="black",          label="agents"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    def XY(arr): 
        return [pos[int(n)][0] for n in arr], [pos[int(n)][1] for n in arr]
    x0, y0 = XY(traces[0])
    scat = ax.scatter(x0, y0, s=40, c="black", alpha=.85, zorder=10)
    ax.set_title("Agent movement (tick=0)")

    def update(t):
        x, y = XY(traces[t]); scat.set_offsets(np.c_[x, y])
        ax.set_title(f"Agent movement (tick={t})"); return scat,

    anim = FuncAnimation(fig, update, frames=len(traces), interval=interval, blit=False)
    writer = PillowWriter(fps=max(1, int(1000/interval)))
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"[Saved GIF] {filename}")


if __name__ == "__main__":
    p = Params()  
    df, sim, A0 = simulate(p)
    print("Anchor fixed at node:", A0)
    print(df.sort_values("footfall", ascending=False).head())

    ensure_dir("figs")
    animate_movement(sim, filename="figs/step2_visible.gif", interval=p.interval_ms)
