"""
定义 Problem 类
"""

import pickle
from .cell_status import CellStatus


class Problem:
    def __init__(
        self,
        res_map: list[list[CellStatus]],
        start: tuple[int, int],
        end: tuple[int, int],
    ):
        """
        初始化 Problem

        :param res_map: 地图矩阵
        :param start: 起点 (i, j)
        :param end: 终点 (i, j)
        """
        # 地图矩阵，其中标记了障碍物(CellStatus.BLOCKED)和无障碍物(CellStatus.EMPTY 或 CellStatus.WALKED)的地方
        self._map = res_map
        # 起点
        self._start = start
        # 终点
        self._end = end
        # 长度和宽度
        self._w = len(res_map[0])
        self._h = len(res_map)

    @classmethod
    def from_file(cls, file_path: str) -> "Problem":
        """
        从文件中读取问题

        (警告，请只加载值得信任的文件，pickle 有代码执行漏洞)

        :param file_path: 文件路径（pickle 序列化文件）
        :return: Problem 对象
        """
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except pickle.PickleError as e:
            raise Exception("Failed to unpickle problem from file: ", e)

    @classmethod
    def from_matrix(cls, matrix: list[list[int]]) -> "Problem":
        """
        从一个矩阵中读入问题。

        :param matrix: 矩阵，0 为空，1 为阻塞，2 为走过，3 为起点，4 为终点
        :return: Problem 对象
        """
        res_map = [[CellStatus.EMPTY] * len(matrix[0]) for _ in range(len(matrix))]
        start_pos = (0, 0)
        end_pos = (len(matrix) - 1, len(matrix[0]) - 1)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 1:
                    res_map[i][j] = CellStatus.BLOCKED
                elif matrix[i][j] == 2:
                    res_map[i][j] = CellStatus.WALKED
                elif matrix[i][j] == 3:
                    start_pos = (i, j)
                elif matrix[i][j] == 4:
                    end_pos = (i, j)

        return Problem(res_map, start_pos, end_pos)

    @property
    def bin_map(self) -> list[list]:
        """
        返回二进制矩阵，0 - 空或者走过，1 - 阻塞

        :return: 二进制矩阵
        """
        b_map = [[0] * len(self._map[0]) for _ in range(len(self._map))]
        for i in range(len(self._map)):
            for j in range(len(self._map[i])):
                if self._map[i][j] == CellStatus.BLOCKED:
                    b_map[i][j] = 1
        return b_map

    def get_numeric_map(self, include_walked=True) -> list[list]:
        """
        返回以数字进行标记的地图矩阵

        :param include_walked: 是否包含走过的路径，如果不包含的话，走过的地方也会被标记为空 (0)
        :return: 地图矩阵，0 为空，1 为阻塞，2 为走过，3 为起点，4 为终点
        """
        n_map = [[0] * len(self._map[0]) for _ in range(len(self._map))]
        for i in range(len(self._map)):
            for j in range(len(self._map[i])):
                if (i, j) == self._start:
                    n_map[i][j] = 3
                elif (i, j) == self._end:
                    n_map[i][j] = 4
                elif self._map[i][j] == CellStatus.BLOCKED:
                    n_map[i][j] = 1
                elif self._map[i][j] == CellStatus.WALKED and include_walked:
                    n_map[i][j] = 2
        return n_map

    @property
    def numeric_map(self):
        """
        (alias) 返回以数字进行标记的地图矩阵，包含走过的路径

        :return: 地图矩阵，0 为空，1 为阻塞，2 为走过，3 为起点，4 为终点
        """
        return self.get_numeric_map()

    @property
    def start(self) -> tuple[int, int]:
        """
        起点

        :return: 起点 (i,j)
        """
        return self._start

    @property
    def end(self) -> tuple[int, int]:
        """
        终点

        :return: 终点 (i,j)
        """
        return self._end

    @property
    def width(self) -> int:
        """
        地图宽度

        :return: 地图宽度
        """
        return self._w

    @property
    def height(self) -> int:
        """
        地图高度

        :return: 地图高度
        """
        return self._h

    def dist_to_end(self, i, j) -> float:
        """
        计算 (i,j) 到终点的预估距离

        :return: 距离
        :note: 实现为欧几里得距离（L2）
        """
        return ((i - self._end[0]) ** 2 + (j - self._end[1]) ** 2) ** 0.5

    def in_bounds(self, i, j) -> bool:
        """
        检查 (i,j) 是否在边界内

        :return: 是否在边界内
        """
        return 0 <= i < self._h and 0 <= j < self._w

    def is_obstacle(self, i, j) -> bool:
        """
        检查 (i,j) 这个地方是不是障碍物。
        （如果 (i,j) 越界，会返回 False，因为越界的地方障碍物也没定义）

        :return: 是否是障碍物
        """
        if not self.in_bounds(i, j):
            return False
        return self._map[i][j] == CellStatus.BLOCKED

    def is_blocked(self, i, j) -> bool:
        """
        检查 (i,j) 这个地方是否走不通

        :return: 是否有障碍物或者越出边界
        :note: 如果 (i,j) 越出边界，会直接返回 True
        """
        if not self.in_bounds(i, j):
            return True
        return self._map[i][j] == CellStatus.BLOCKED

    def save(self, file_path: str):
        """
        把 Problem 对象持久化存储到路径 file_path

        （实现用的是 pickle）

        :param file_path: 文件路径
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

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
