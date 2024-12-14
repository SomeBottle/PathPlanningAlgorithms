"""
制作问题过程中能用到的工具函数
"""

import numpy as np
from .cell_status import CellStatus


def close_problem_obstacles(
    res_map: list[list[CellStatus]],
) -> list[list[CellStatus]]:
    """
    对问题障碍物矩阵进行处理，封闭掉一些对角障碍物，消除对角障碍物问题。

    :param res_map: 问题障碍物矩阵
    :return: 处理后的障碍物矩阵
    """
    # 复制后转换为 0/1 np 矩阵
    map_copy = [row[:] for row in res_map]
    np_map = np.array(
        [
            [
                0 if ele == CellStatus.EMPTY or ele == CellStatus.WALKED else 1
                for ele in row
            ]
            for row in res_map
        ]
    )
    # 封闭一些障碍物位置，防止可以向八个方向移动时，可以穿过角落
    """
    类似这种情况：

    #    或    #
      #      #
    """
    # 窗口大小为 2x2，扫描整张矩阵
    # 2x2 对角阵
    diag_unit = np.diag([1, 1])  # 主对角
    counter_diag_unit = np.fliplr(diag_unit)  # 副对角
    for i in range(np_map.shape[0] - 1):
        for j in range(np_map.shape[1] - 1):
            if np.array_equal(np_map[i : i + 2, j : j + 2], diag_unit):
                # 主对角，填左下角
                map_copy[i + 1][j] = CellStatus.BLOCKED
            elif np.array_equal(np_map[i : i + 2, j : j + 2], counter_diag_unit):
                # 副对角，填右下角
                map_copy[i + 1][j + 1] = CellStatus.BLOCKED

    return map_copy
