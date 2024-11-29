from enum import Enum


class AlgorithmState(Enum):
    INITIALIZED = 0  # 已经初始化
    RUNNING = 1  # 正在进行
    SOLVED = 2  # 已经解决
    ENDED = 3  # 没有解决但算法已经终止
