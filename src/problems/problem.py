"""
定义 Problem 类
"""

from .cell_status import CellStatus


class Problem:
    def __init__(
        self, res_map: list[list], start: tuple[int, int], end: tuple[int, int]
    ):
        """
        初始化 Problem

        :param res_map: 地图矩阵
        :param start: 起点 (i, j)
        :param end: 终点 (i, j)
        """
        # 地图矩阵，其中标记了障碍物(1)和无障碍物(0)的地方
        self._map = res_map
        # 起点
        self._start = start
        # 终点
        self._end = end

    def __str__(self):
        res_str = ""
        for i in range(len(self._map)):
            for j in range(len(self._map[0])):
                if (i, j) == self._start:
                    res_str += "S"
                    continue
                elif (i, j) == self._end:
                    res_str += "E"
                    continue

                if self._map[i][j] == CellStatus.BLOCKED:
                    res_str += "#"
                elif self._map[i][j] == CellStatus.WALKED:
                    res_str += "."
                else:
                    res_str += " "
            res_str += "\n"
        return res_str
