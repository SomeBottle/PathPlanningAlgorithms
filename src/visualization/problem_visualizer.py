import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from problems import Problem
from .utils import scaled_figsize, hex_to_rgb

# 必须是 16 进制字符串
COLORS = ["#FFFFFF", "#000000", "#A8CD89", "#D91656", "#80C4E9"]


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


def visualize_as_color_matrix(problem: Problem) -> list[list[tuple[int, int, int]]]:
    """
    将问题可视化为颜色矩阵

    :return: 返回一个 RGB 颜色矩阵
    """
    matrix = problem.numeric_map
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            color = COLORS[matrix[i][j]]
            # 转换为颜色矩阵
            matrix[i][j] = hex_to_rgb(color)
    return matrix
