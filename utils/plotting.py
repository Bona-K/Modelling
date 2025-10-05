import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import networkx as nx

def set_palette(style = 'whitegrid', context = 'talk', palette = 'Set2'):
    sns.set_theme(style=style, context=context, palette=palette)
    print(f"[Seaborn] style={style}, context={context}, palette={palette}")

def three_frame(dfs, steps_list, title="Footfall comparison", figsize=(14,4)):
    sns.set_theme(style="whitegrid", palette="Set2")
    fig, axes = plt.subplots(1, len(dfs), figsize=figsize, sharey=True)

    for ax, df, steps in zip(axes, dfs, steps_list):
        sns.barplot(data=df.sort_values("node"), x="node", y="footfall", hue="category", ax=ax)
        ax.set_title(f"{steps} steps")
        ax.set_xlabel("Node"); ax.set_ylabel("Footfall")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def animate_movement(sim, filename="data/outputs/mall_walk.gif", pos=None, interval=400):
    """
    Animate agent movement through the shopping mall network.

    Parameters
    ----------
    sim : object
        Simulation object with attributes:
            - mall_graph : networkx.Graph
            - traces : list of arrays/lists (each tick's agent positions)
            - p.seed : int (for reproducible layout)
    filename : str
        Path to save GIF animation.
    pos : dict, optional
        Node positions (if None, spring_layout is generated)
    interval : int
        Frame interval (ms)

    Notes
    -----
    Node color mapping:
        - anchor     → red
        - similar    → blue
        - different  → green
    """

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    import os

    G = sim.mall_graph
    traces = sim.traces

    if pos is None:
        pos = nx.spring_layout(G, seed=sim.p.seed, k=1.0/np.sqrt(G.number_of_nodes()))

    color_map = {
        "anchor": "tab:red",
        "similar": "tab:blue",
        "different": "tab:green"
    }

    node_colors = []
    for n in G.nodes:
        role = G.nodes[n].get("role", "")
        cat = G.nodes[n].get("category", "")
        if role == "anchor":
            node_colors.append(color_map["anchor"])
        else:
            node_colors.append(color_map.get(cat, "gray"))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=180, ax=ax)

    # legend
    handles = [
        plt.Line2D([0],[0], marker="o", ls='', color=color_map["anchor"], label="anchor"),
        plt.Line2D([0],[0], marker="o", ls='', color=color_map["similar"], label="similar"),
        plt.Line2D([0],[0], marker="o", ls='', color=color_map["different"], label="different")
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    first = traces[0]
    xs = [pos[int(n)][0] for n in first]
    ys = [pos[int(n)][1] for n in first]
    scat = ax.scatter(xs, ys, s=18, alpha=0.7, color="black")
    ax.set_title("Agent movement (tick=0)")

    def update(frame):
        arr = traces[frame]
        xs = [pos[int(n)][0] for n in arr]
        ys = [pos[int(n)][1] for n in arr]
        scat.set_offsets(np.c_[xs, ys])
        ax.set_title(f"Agent movement (tick={frame})")
        return scat,

    anim = FuncAnimation(fig, update, frames=len(traces), interval=interval, blit=True)

    writer = PillowWriter(fps=max(1, int(1000/interval)))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"[Saved GIF] {filename}")
