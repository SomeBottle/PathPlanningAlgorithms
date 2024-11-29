import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from problems import Problem
from .utils import scaled_figsize


def visualize(problem: Problem, path: list[tuple[int, int]]):
    """
    可视化结果路径

    :param problem: 问题
    :param path: 路径（坐标组成）
    """
    # 在 numeric_map 的基础上处理
    w = problem.width
    h = problem.height
    # 计算宽高比例确定图表大小
    w, h = scaled_figsize(w, h)
    fig, ax = plt.subplots(figsize=(w, h))
