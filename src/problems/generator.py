"""
寻路问题生成器
"""

import random
import math

from .problem import Problem
from .cell_status import CellStatus
from .utils import close_problem_obstacles
from exceptions import InvalidProblemArgument

# 图像短边至少要有多长
MIN_SIDE_LENGTH = 50


def _rand_range(range_size: int) -> tuple:
    """
    在 [0, range_size) 区间内随机取出一段区间

    :param range_size: 区间大小
    :return: 随机取出的区间
    """
    start = random.randrange(range_size)
    end = random.randrange(start, range_size)
    return (start, end)


def _check_available(res_map: list[list], i: int, j: int) -> bool:
    """
    检查地图上 (i,j) 处是否能走

    :param res_map: 地图矩阵
    :param i: 行号
    :param j: 列号
    :return: 是否能走
    """
    if i < 0 or i >= len(res_map) or j < 0 or j >= len(res_map[0]):
        return False
    return res_map[i][j] != CellStatus.WALKED


def _gen_random_start_end(
    width: int, height: int, distance: int | None = None
) -> tuple[int, int, int, int]:
    """
    随机生成开始和终止点的坐标

    :param width: 地图宽度
    :param height: 地图高度
    :param distance: 起点到终点至少要满足的 L2 距离，为 None 时会固定起点在左上角，终点在右下角。
    :return: 起点和终点的坐标 (start_i, start_j, end_i, end_j)
    """
    if distance is not None:
        while True:
            start_i, start_j = random.randrange(height), random.randrange(width)
            end_i, end_j = random.randrange(height), random.randrange(width)
            if (
                ((start_i - end_i) ** 2 + (start_j - end_j) ** 2) >= distance**2
                and start_i != end_i
                and start_j != end_j
            ):
                break
    else:
        # distance 为 None 则固定起点在左上角，终点在右下角
        start_i, start_j = 0, 0
        end_i, end_j = height - 1, width - 1

    return (start_i, start_j, end_i, end_j)


def _float_range(start, end, step):
    """
    生成一个浮点数范围

    :param start: 起始值
    :param end: 结束值
    :param step: 步长
    :return: 浮点数范围
    """
    while start < end:
        yield start
        start += step


def generate_partial_ring_problem(
    width: int, height: int, ring_num: int = 1, distance: int | None = None
) -> Problem:
    """
    生成有半环型障碍的问题，生成的问题中不包含走过的路径（WALKED）

    :param width: 地图宽度
    :param height: 地图高度
    :param ring_num: 半环障碍的个数
    :param distance: 起点到终点至少要满足的 L2 距离，为 None 时会固定起点在左上角，终点在右下角。
    :return: 生成的寻路问题 Problem 对象
    """

    # width, height 太小
    if min(width, height) < MIN_SIDE_LENGTH:
        raise ValueError(f"width and height should be at lease {MIN_SIDE_LENGTH}")

    if distance is not None:
        max_l2_dist = width**2 + height**2
        # 判断输入的 distance 是否有可能满足
        if max_l2_dist < (distance**2) * 1.2:
            raise InvalidProblemArgument(f"Invalid distance: {distance}, unsolvable.")

    start_i, start_j, end_i, end_j = _gen_random_start_end(width, height, distance)

    # 最初整张图是空图
    res_map = [[CellStatus.EMPTY for _ in range(width)] for _ in range(height)]

    # 起点和终点标记一下
    res_map[start_i][start_j] = CellStatus.WALKED
    res_map[end_i][end_j] = CellStatus.WALKED

    # 计算起点和终点之间的 L2 距离
    l2_dist = math.sqrt((start_i - end_i) ** 2 + (start_j - end_j) ** 2)

    # 如果距离太短，也认为不可解
    if l2_dist < 2:
        raise InvalidProblemArgument(f"width and height too small.")

    # 随机选择起点或者终点为圆心
    if random.random() < 0.5:
        center_i, center_j = start_i, start_j
    else:
        center_i, center_j = end_i, end_j

    # 生成半环障碍
    for _ in range(ring_num):
        radius = random.uniform(l2_dist / 6, l2_dist * 5 / 6)
        # 确定水平方向偏移的范围
        # 最大的水平偏移绝对值为 radius
        x_offset_range: tuple[int, int] = (
            math.ceil(max(0, center_j - radius)),
            math.floor(min(width - 1, center_j + radius)),
        )

        # 绘制方向决定了半环在圆心的方向
        direction = random.choice(
            [
                range(x_offset_range[0], x_offset_range[1] + 1),
                range(x_offset_range[1], x_offset_range[0] - 1, -1),
            ]
        )

        x_list = list(direction)
        #print(f"Center: ({center_i},{center_j}), xrange: {x_list}, radius:{radius}")
        prev_x = None  # 记录上一个 x，用于插值，把圆填的圆满一点
        for x in direction:
            # 比如 x = upper_end 时就会停止绘制上半部分
            # 如果有 prev_x，计算插值，让半环能封闭
            if prev_x != None:
                sign = 1 if x - prev_x > 0 else -1  # 计算符号
                interporlated_range = _float_range(prev_x, x, 0.1 * sign)
            else:
                interporlated_range = range(x, x + 1)
            for int_x in interporlated_range:
                # 计算垂直方向上的偏移
                y_off = round(math.sqrt(radius**2 - (int_x - center_j) ** 2))
                # 计算对应 int_x 的，在圆心上方和下方的 y 坐标
                top_y = center_i + y_off
                bottom_y = center_i - y_off

                if _check_available(res_map, top_y, x):
                    res_map[top_y][x] = CellStatus.BLOCKED

                if _check_available(res_map, bottom_y, x):
                    res_map[bottom_y][x] = CellStatus.BLOCKED

            prev_x = x

    # 填充对角处的障碍物，以支持搜索过程中 8 个方向的扩展
    res_map = close_problem_obstacles(res_map)

    return Problem(res_map, (start_i, start_j), (end_i, end_j))


def generate_random_problem(
    width: int,
    height: int,
    available_cell_percent: float,
    distance: int | None = None,
    noise_strength: float = 0.95,
) -> Problem:
    """
    随机生成一个寻路问题，这个方法生成的问题中包含有生成时的过程路径（WALKED）。

    :param width: 地图宽度
    :param height: 地图高度
    :param available_cell_percent: 可行走区域至少占的比例
    :param distance: 起点到终点至少要满足的 L2 距离，为 None 时会固定起点在左上角，终点在右下角。
    :param noise_strength: 噪音强度（0-1），噪音越强路线抖动越厉害
    :return: 生成的寻路问题 Problem 对象
    """

    # width, height 太小
    if min(width, height) < MIN_SIDE_LENGTH:
        raise ValueError(f"width and height should be at lease {MIN_SIDE_LENGTH}")

    area = width * height
    # 需要有多少格子有空位
    available_cells = area * available_cell_percent
    if distance is not None:
        max_l2_dist = width**2 + height**2
        # 从欧几里得和曼哈顿距离来估计输入的参数是否能满足
        if (
            max_l2_dist < (distance**2) * 1.2
            or area - (available_cells - width - height) <= 0
        ):
            # 距离约束是无法满足的
            raise InvalidProblemArgument(f"Invalid distance: {distance}, unsolvable.")

    start_i, start_j, end_i, end_j = _gen_random_start_end(width, height, distance)

    # 先把整张图填充一下
    res_map = [[CellStatus.BLOCKED for _ in range(width)] for _ in range(height)]

    # 先打通起点到终点的一条路径，用曼哈顿距离
    i, j = start_i, start_j
    res_map[i][j] = CellStatus.WALKED
    # 空出来的格子数
    empty_cell_count = 1
    while i != end_i or j != end_j:
        # 计算方向
        dist_i = end_i - i
        dist_j = end_j - j
        abs_dist_i, abs_dist_j = abs(dist_i), abs(dist_j)
        l_1_dist = abs_dist_i + abs_dist_j
        # 转换为单位向量表示方向
        d_i = dist_i // abs_dist_i if dist_i != 0 else 0
        d_j = dist_j // abs_dist_j if dist_j != 0 else 0
        # 限制不能斜着走，d_i 和 d_j 只有一个可以是 1
        if d_i + d_j != -1 and d_i + d_j != 1:
            # 并不是只有一个为 1，随机把一个置 0
            if random.random() < 1 - (
                min(abs_dist_i, abs_dist_j) / max(abs_dist_i, abs_dist_j)
            ):
                # 动态计算高度方向和宽度方向置零的概率
                # 如果短距离比长距离差的更多，就优先把长距离变化量置零
                if abs_dist_i > abs_dist_j:
                    d_i = 0
                else:
                    d_j = 0
            else:
                if abs_dist_i > abs_dist_j:
                    d_j = 0
                else:
                    d_i = 0

        # 从 start 到 end 的路径中会以 p 的概率改变方向（抖动）
        # p 根据 noise_strength 和距离终点的曼哈顿距离来确定
        p = math.exp(
            -(end_i - start_i + end_j - start_j - l_1_dist)
            * (1 / (end_j - start_j))
            * (1 - noise_strength)
        )

        if random.random() < p:
            # 噪声：和当前方向垂直的一个可行方向
            m_i, m_j = d_j, d_i
            if _check_available(res_map, i + m_i, j + m_j):
                d_i = m_i
                d_j = m_j
            elif _check_available(res_map, i - m_i, j - m_j):
                d_i = -m_i
                d_j = -m_j

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
        action = random.randint(0, 1)
        if action == 0:
            chosen_i = random.randrange(height)
            for j in range(*_rand_range(width)):
                if res_map[chosen_i][j] == CellStatus.BLOCKED and random.gauss() > 1:
                    res_map[chosen_i][j] = CellStatus.EMPTY
                    empty_cell_count += 1
        else:
            # 再随机选择一列，把其中某些被阻塞的格子置空
            chosen_j = random.randrange(width)
            for i in range(*_rand_range(height)):
                if res_map[i][chosen_j] == CellStatus.BLOCKED and random.gauss() > 1:
                    res_map[i][chosen_j] = CellStatus.EMPTY
                    empty_cell_count += 1

    return Problem(res_map, (start_i, start_j), (end_i, end_j))
