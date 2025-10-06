# utils/plotting.py
import matplotlib.pyplot as plt

def bar_dict(d: dict, title: str, xlabel: str, ylabel: str):
    xs = list(d.keys())
    ys = list(d.values())
    plt.figure()
    plt.bar([str(x) for x in xs], ys)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()

def line_dict(d: dict, title: str, xlabel: str, ylabel: str):
    xs = list(d.keys())
    ys = list(d.values())
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
