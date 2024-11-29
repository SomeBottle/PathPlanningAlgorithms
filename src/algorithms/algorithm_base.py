"""
寻路算法基类
"""


class AlgorithmBase:

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

    def solve(self):
        """
        执行算法直至算法终止
        """
        pass
