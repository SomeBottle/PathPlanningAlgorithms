"""
辅助算法执行的一些类和函数
"""


# 表示各个方向 (di, dj)
class Direction:
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP = (-1, 0)
    DOWN = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)

    @classmethod
    def get(
        cls, coordinate_1: tuple[int, int], coordinate_2: tuple[int, int]
    ) -> tuple[int, int]:
        """
        计算从 coordinate_1 到 coordinate_2 的方向

        :param coordinate_1: 运动前的坐标
        :param coordinate_2: 运动后的坐标
        :return: 方向 (di, dj), di,dj ∈ {-1,0,1}
        """
        di = coordinate_2[0] - coordinate_1[0]
        dj = coordinate_2[1] - coordinate_1[1]
        return (
            int(di / abs(di)) if di != 0 else 0,
            int(dj / abs(dj)) if dj != 0 else 0,
        )

    @classmethod
    def dist(
        cls, coordinate_1: tuple[int, int], coordinate_2: tuple[int, int]
    ) -> float:
        """
        计算从 coordinate_1 到 coordinate_2 的欧几里得距离  

        :param coordinate_1: 运动前的坐标
        :param coordinate_2: 运动后的坐标
        :return: 欧几里得距离
        """
        return (
            (coordinate_1[0] - coordinate_2[0]) ** 2
            + (coordinate_1[1] - coordinate_2[1]) ** 2
        ) ** 0.5

    @classmethod
    def step(
        cls, coordinate: tuple[int, int], direction: tuple[int, int]
    ) -> tuple[int, int]:
        """
        从 coordinate 开始在方向 direction 上走一步后的新坐标

        :param coordinate: 当前坐标
        :param direction: 方向
        :return: 新坐标
        """
        return (coordinate[0] + direction[0], coordinate[1] + direction[1])

    @classmethod
    def all(cls) -> tuple[tuple[int, int], ...]:
        """
        返回所有的方向
        """
        return (
            cls.LEFT,
            cls.RIGHT,
            cls.UP,
            cls.DOWN,
            cls.UP_LEFT,
            cls.UP_RIGHT,
            cls.DOWN_LEFT,
            cls.DOWN_RIGHT,
        )

    @classmethod
    def orthogonal(
        cls, direction: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        返回 direction 方向的正交方向

        :param direction: 方向
        :return: 正交方向，有两个
        """
        return (
            (-direction[1], direction[0]),
            (direction[1], -direction[0]),
        )

    @classmethod
    def is_diagonal(cls, direction: tuple[int, int]) -> bool:
        """
        判断 direction 是否是斜对角方向

        :param direction: 方向
        :return: 是否是斜对角方向
        """
        return direction[0] != 0 and direction[1] != 0
