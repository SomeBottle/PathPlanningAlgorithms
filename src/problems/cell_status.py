from enum import Enum


class CellStatus(Enum):
    EMPTY = 0  # 未走过
    BLOCKED = 1  # 障碍
    WALKED = 2  # 走过
