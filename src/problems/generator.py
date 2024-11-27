"""
寻路问题生成器
"""

import random

from .problem import Problem
from .cell_status import CellStatus
from exceptions import InvalidProblemArgument


# 最初的 P
INITIAL_P = 1


def _rand_range(range_size: int) -> tuple:
    """
    在 [0, range_size) 区间内随机取出一段区间

    :param range_size: 区间大小
    :return: 随机取出的区间
    """
    start = random.randrange(range_size)
    end = random.randrange(start, range_size)
    return (start, end)


def _check_available(res_map: int, i: int, j: int) -> bool:
    """
    检查地图上 (i,j) 处是否能走

    :param res_map: 地图矩阵
    :param i: 行号
    :param j: 列号
    :return: 是否能走
    """
    if i < 0 or i >= len(res_map) or j < 0 or j >= len(res_map[0]):
        return False
    return res_map[i][j] == CellStatus.EMPTY


def generate_problem(
    width: int,
    height: int,
    distance: int,
    available_cell_percent: float,
    p_decay_factor: float = 0.99,
) -> Problem:
    """
    随机生成一个寻路问题

    :param width: 地图宽度
    :param height: 地图高度
    :param distance: 起点到终点至少要满足的 L2 距离
    :param available_cell_percent: 可行走区域至少占的比例
    :param p_decay_factor: p 衰减系数。以 p 的概率改变方向，衰减系数越大，路线越稳定
    :return: 生成的寻路问题 Problem 对象
    """
    max_l2_dist = width**2 + height**2
    area = width * height
    # 需要有多少格子有空位
    available_cells = area * available_cell_percent
    # 从欧几里得和曼哈顿距离来估计输入的参数是否能满足
    if (
        max_l2_dist < (distance**2) * 1.2
        or area - (available_cells - width - height) <= 0
    ):
        # 距离约束是无法满足的
        raise InvalidProblemArgument(f"Invalid distance: {distance}, unsolvable.")
    # 先把整张图填充一下，有极小概率空出一些地方
    res_map = [[CellStatus.BLOCKED for _ in range(width)] for _ in range(height)]

    # 随机生成起点和终点，直至满足距离要求
    while True:
        start_i, start_j = random.randrange(height), random.randrange(width)
        end_i, end_j = random.randrange(height), random.randrange(width)
        if ((start_i - end_i) ** 2 + (start_j - end_j) ** 2) >= distance**2:
            break

    # 从 start 到 end 的路径中会以 p 的概率改变方向（抖动）
    p = INITIAL_P

    # 先打通起点到终点的一条路径，用曼哈顿距离
    i, j = start_i, start_j
    res_map[i][j] = CellStatus.WALKED
    # 空出来的格子数
    empty_cell_count = 1
    while i != end_i or j != end_j:
        # 计算方向
        d_i = end_i - i
        d_j = end_j - j
        # 转换为单位向量表示方向
        d_i = d_i // abs(d_i) if d_i != 0 else 0
        d_j = d_j // abs(d_j) if d_j != 0 else 0
        # 限制不能斜着走，d_i 和 d_j 只有一个可以是 1
        if d_i + d_j != -1 and d_i + d_j != 1:
            # 并不是只有一个为 1，随机把一个置 0
            if random.random() < 0.5:
                d_i = 0
            else:
                d_j = 0

        # 以 p 的概率改变方向
        if random.random() < p:
            factor_i, factor_j = random.choice([(-1, 1), (1, -1), (-1, -1)])
            m_i = d_i * factor_i
            m_j = d_j * factor_j
            if _check_available(res_map, i + m_i, j + m_j):
                d_i = m_i
                d_j = m_j

            # 衰减
            p *= p_decay_factor

        # 移动
        i += d_i
        j += d_j

        if res_map[i][j] != CellStatus.WALKED:
            # 更新空出来的格子数目
            empty_cell_count += 1
        res_map[i][j] = CellStatus.WALKED

    # 走出一条路线后再随机打通一些地方
    while empty_cell_count < available_cells:
        # 先随机选择一行，把其中某些被阻塞的格子置空
        action = random.randint(0, 2)
        if action == 1:
            chosen_i = random.randrange(height)
            for j in range(*_rand_range(width)):
                if res_map[chosen_i][j] == CellStatus.BLOCKED:
                    res_map[chosen_i][j] = CellStatus.EMPTY
                    empty_cell_count += 1
        elif action == 2:
            # 再随机选择一列，把其中某些被阻塞的格子置空
            chosen_j = random.randrange(width)
            for i in range(*_rand_range(height)):
                if res_map[i][chosen_j] == CellStatus.BLOCKED:
                    res_map[i][chosen_j] = CellStatus.EMPTY
                    empty_cell_count += 1
        else:
            # 以随机概率挖空一些地方
            for i in range(height):
                for j in range(width):
                    if res_map[i][j] == CellStatus.BLOCKED and random.gauss() > 0.5:
                        res_map[i][j] = CellStatus.EMPTY
                        empty_cell_count += 1

    return Problem(res_map, (start_i, start_j), (end_i, end_j))
