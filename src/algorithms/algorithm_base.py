"""
寻路算法基类
"""

from typing import Generator
from problems import Problem
from .states import AlgorithmState


class AlgorithmBase:

    EMPTY_COLOR = "#FFFFFF"  # 空白处的颜色

    def __init__(self, problem: Problem, record_int=False, diagonal_obstacles=True):
        """
        初始化算法

        :param: problem: 寻路问题对象
        :param: record_int: 是否记录中间结果，中间结果是用于 next_visual_generator 进行可视化动画展示的
        :param diagonal_obstacles: 是否把对角障碍物考虑在内。搜索方向为对角时可能遇到对角障碍物，即四个格子中有其中一个对角存在障碍物，另一个对角是空的的情况。
        """
        pass

    def has_next_step(self) -> bool:
        """
        算法是否还有下一步

        :return: 是否还有下一步
        """
        pass

    def next_step(self) -> bool:
        """
        执行算法的下一步

        :return: 本步是否完成执行
        """
        pass

    def next_visual_generator(
        self,
    ) -> Generator[list[list[tuple[int, int, int]]], None, None]:
        """
        调用算法 next_step 并产生可视化结果的生成器

        :return: 可视化结果 Generator，产生一个矩阵，包含每个位置的颜色的 RGB 值 (R,G,B)
        """

    def solve(self):
        """
        执行算法直至算法终止
        """
        pass

    @property
    def problem(self) -> Problem:
        """
        算法求解的原始问题
        """
        pass

    @property
    def state(self) -> AlgorithmState:
        """
        获得算法的状态

        :return: 算法的状态
        """
        pass
