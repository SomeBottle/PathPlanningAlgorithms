import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from problems import Problem
from .utils import scaled_figsize

COLORS = ["white", "black", "#A59D84", "#D91656", "#80C4E9"]


def visualize(problem: Problem, path: list[tuple[int, int]]):
    """
    可视化结果路径

    :param problem: 问题
    :param path: 路径（坐标组成）
    """
    # 转化为集合，方便查找
    path = set(path)
    # 在 numeric_map 的基础上处理
    res_map = problem.get_numeric_map(include_walked=False)
    for i in range(len(res_map)):
        for j in range(len(res_map[0])):
            if (i, j) in path and res_map[i][j] != 3 and res_map[i][j] != 4:
                # 标记走过的路径
                res_map[i][j] = 2  # 2 代表走过的地方
    w = problem.width
    h = problem.height
    # 计算宽高比例确定图表大小
    w, h = scaled_figsize(w, h)
    fig, ax = plt.subplots(figsize=(w, h))
    cmap = ListedColormap(COLORS)
    ax.imshow(res_map, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    plt.axis("off")
    plt.show()
