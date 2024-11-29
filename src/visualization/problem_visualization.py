import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from problems import Problem


COLORS = ["white", "black", "#A8CD89", "#D91656", "#80C4E9"]

# Matplotlib 图表中短边的长度（inch）
SHORTER_SIDE_LEN = 6


def _scale_fig(m: float, n: float) -> tuple[float, float]:
    """
    按比例缩放，直至 m,n 中较小的一个到 SHORTER_SIDE_LEN

    :return: 缩放后的数值 (m,n)
    """
    smaller = min(m, n)
    factor = smaller / SHORTER_SIDE_LEN
    m /= factor
    n /= factor
    return (m, n)


def visualize(problem: Problem):
    """
    将问题可视化展示出来
    """
    w = problem.width
    h = problem.height
    # 计算宽高比例确定图表大小
    w, h = _scale_fig(w, h)
    fig, ax = plt.subplots(figsize=(w, h))
    cmap = ListedColormap(COLORS)
    ax.imshow(problem.numeric_map, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    plt.axis("off")
    plt.show()
