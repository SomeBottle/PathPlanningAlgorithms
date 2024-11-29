import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from problems import Problem
from .utils import scaled_figsize


COLORS = ["white", "black", "#A8CD89", "#D91656", "#80C4E9"]


def visualize(problem: Problem):
    """
    将问题可视化展示出来
    """
    w = problem.width
    h = problem.height
    # 计算宽高比例确定图表大小
    w, h = scaled_figsize(w, h)
    fig, ax = plt.subplots(figsize=(w, h))
    cmap = ListedColormap(COLORS)
    ax.imshow(problem.numeric_map, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    plt.axis("off")
    plt.show()
