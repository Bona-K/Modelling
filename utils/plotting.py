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
    
    plt.suptilte(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def animate_movement(sim, filename="data/outputs/mall_walk.gif", pos=None, interval=400):
    G = sim.mall_graph
    traces = sim.traces
    if pos is None:
        pos = nx.spring_layout(G, seed=sim.p.seed, k=1.0/np.sprt(G.number_of_nodes()))
    
    color_map = {"origin":"tab:red", "similar":"tab:blue", "different":"tab:green"}
    node_colors = [color_map.get(G.nodes[n]["category"], "gray") for n in G.nodes]

    fig, ax = plt.subplots(figsize=(7,6)); ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=180, ax=ax)

    handles = [plt.Line2D([0],[0], marker="o", ls='', color=color_map[k], label=k)
               for k in ["origin","similar","different"]]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    first = traces[0]
    xs = [pos[int(n)][0] for n in first]; ys = [pos[int(n)][1] for n in first]
    scat = ax.scatter(xs, ys, s=18, alpha=0.7)
    ax.set_title("Agent movement (tick=0)")

    def update(frame):
        arr = traces[frame]
        xs = [pos[int(n)][0] for n in arr]; ys = [pos[int(n)][0] for n in arr]
        scat.set_offsets(np.c_[xs, ys])
        ax.set_title(f"Agent movement (tick={frame})")
        return scat,

    anim = FuncAnimation(fig, update, frames=len(traces), interval=interval, blit=True)
    writer = PillowWriter(fps=max(1, int(1000/interval)))
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"[Saved GIF] {filename}")
