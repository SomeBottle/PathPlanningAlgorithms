"""
A* 算法（带堆优化）
"""

import heapq

from problems import Problem
from .states import AlgorithmState
from .algorithm_base import AlgorithmBase

# 这里如果从 visualization 导入会造成环型导入
from visualization.utils import hex_to_rgb
from typing import Generator

# 规定可行的方向
# DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# 八个方向
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

# 用于展示中间结果
CLOSED_COLOR = "#A59D84"  # 确定下来的路径的颜色
OPEN_COLOR = "#FFD2A0"  # 待探索的路径的颜色
NEIGHBOR_COLOR = "#85A98F"  # 每个位置邻居的颜色
UPDATED_NEIGHBOR_COLOR = "#D91656"  # 更新了的邻居的颜色
PATH_COLOR = "#D91656"  # 走过的路径的颜色


# A* 算法的结点
class AStarNode:
    def __init__(
        self, path_cost=0.0, dist_to_end=0.0, parent=None, pos=(0, 0), record_int=False
    ):
        # 这个结点距离起点的路径的代价
        self.path_cost: float = path_cost
        # 这个结点距离终点的预估代价
        self.dist_to_end: float = dist_to_end
        # 这个结点的父节点，用于找到路径
        self.parent: AStarNode | None = parent
        # 这个结点的位置
        self.pos: tuple[int, int] = pos

    def __lt__(self, other):
        """
        主要用于实现小根堆，heapq 应该是用 '<' 进行比较的

        :note: 算法每一次都要从 open_list 中找到距离起点和终点的距离之和最小的结点
        """
        return self.path_cost + self.dist_to_end < other.path_cost + other.dist_to_end


# A* 算法
class AStarAlgorithm(AlgorithmBase):

    def __init__(self, problem: Problem, record_int=False):
        super().__init__(problem, record_int)
        # Open List 实际上是一个小根堆
        self._open_list = []
        # Open Dict 存储 (i,j) -> AStarNode 的映射
        self._open_dict = {}
        # Closed Dict 存储 (i,j) -> AStarNode 的映射
        self._closed_dict = {}
        self._problem = problem
        # 记录最终的路径
        self._solution_path: list[AStarNode] = []
        # ============== 存储中间数据初始化 ==============
        # 是否存储中间数据
        self._record_int = record_int

        if record_int:
            # 存储每个像素绘制什么颜色
            self._int_matrix = [
                [self.EMPTY_COLOR] * self._problem.width
                for _ in range(self._problem.height)
            ]
            # 存储邻居的位置
            self._neighbors: list[tuple[int, int]] = []
            # 存储发生更新的邻居的位置
            self._updated_neighbors: list[tuple[int, int]] = []
        # ============== 存储中间数据初始化完成 ==============

        # 把起点加入到开放列表中
        self._add_as_open(AStarNode(pos=problem.start))
        # 算法状态
        self._state: AlgorithmState = AlgorithmState.INITIALIZED

    @property
    def problem(self):
        return self._problem

    def _add_as_open(self, node: AStarNode):
        """
        将结点加入到开放列表中

        :param node: 要加入的结点
        """
        heapq.heappush(self._open_list, node)
        self._open_dict[node.pos] = node

        # =========== 更新中间数据 ===========
        if self._record_int:
            self._int_matrix[node.pos[0]][node.pos[1]] = OPEN_COLOR
        # =========== 中间数据更新完成 ===========

    def _pop_min_open(self) -> AStarNode:
        """
        弹出开放列表中代价最小的结点

        :return: 弹出的结点
        """
        node: AStarNode = heapq.heappop(self._open_list)
        # 同时从 dict 中移除
        del self._open_dict[node.pos]
        return node

    def _cache_solution(self, end_node: AStarNode):
        """
        从最后一个结点，通过父结点开始逆推，得到最终的结果路径

        :param end_node: 最后一个结点
        """
        self._solution_path = []
        while end_node is not None:
            self._solution_path.append(end_node)
            end_node = end_node.parent
        self._solution_path.reverse()

    def has_next_step(self) -> bool:
        if len(self._open_list) == 0:
            # 开放列表中已经空了
            self._state = AlgorithmState.ENDED

        if self._state in (AlgorithmState.SOLVED, AlgorithmState.ENDED):
            # 算法已经结束就不能继续了
            return False

        return True

    def next_step(self) -> bool:
        if not self.has_next_step():
            # 没有下一步了
            return False

        # 取得到起点和终点距离之和最小的结点
        curr_node: AStarNode = self._pop_min_open()
        # 标记此结点已经确定（加入 Closed Dict）
        self._closed_dict[curr_node.pos] = curr_node

        # =========== 更新中间数据 ===========
        if self._record_int:
            self._int_matrix[curr_node.pos[0]][curr_node.pos[1]] = CLOSED_COLOR
            # 寻找本次邻居前清空之前的邻居数据
            self._neighbors.clear()
            self._updated_neighbors.clear()
            # 生成中间路径
            self._cache_solution(curr_node)
        # =========== 中间数据更新完成 ===========

        # 检查是不是终点
        if curr_node.pos == self._problem.end:
            # 到达终点
            self._state = AlgorithmState.SOLVED
            # 生成最终路径
            self._cache_solution(curr_node)
            return True

        # 寻找邻居，将可行的地方加入到 open_list
        for deltas in DIRECTIONS:
            # 邻居的坐标
            new_pos = (curr_node.pos[0] + deltas[0], curr_node.pos[1] + deltas[1])
            if not self._problem.in_bounds(*new_pos) or self._problem.is_blocked(
                *new_pos
            ):
                # 如果这个邻居格子不可行或者是障碍物，就跳过
                continue
            if new_pos in self._closed_dict:
                # 如果已经访问过并确定下来了，也跳过
                continue

            # 新的路径代价（邻居距离终点的预估代价是没有变的）
            new_path_cost = curr_node.path_cost + 1  # 相距一步

            # =========== 更新中间数据 ===========
            if self._record_int:
                self._neighbors.append(new_pos)  # 记录邻居
            # =========== 中间数据更新完成 ===========

            # 判断这些邻居之前有没有被加入过 open_list
            if new_pos in self._open_dict:
                # 如果加入过，看看能不能更新路径
                neighbor_node: AStarNode = self._open_dict[new_pos]
                if new_path_cost < neighbor_node.path_cost:
                    # 如果从当前结点到这个邻居代价更小，则更新
                    neighbor_node.parent = curr_node
                    neighbor_node.path_cost = new_path_cost
                    # 因为小根堆的性质，更新路径后需要重新排序
                    heapq.heapify(self._open_list)

                    # =========== 更新中间数据 ===========
                    if self._record_int:
                        self._updated_neighbors.append(
                            new_pos
                        )  # 记录被更新了路径的邻居
                    # =========== 中间数据更新完成 ===========
            else:
                # 否则把邻居加入 open_list
                neighbor_node: AStarNode = AStarNode(
                    path_cost=new_path_cost,
                    parent=curr_node,
                    pos=new_pos,
                    dist_to_end=self._problem.dist_to_end(*new_pos),
                )
                self._add_as_open(neighbor_node)

        return True

    def next_visual_generator(
        self,
    ) -> Generator[list[list[tuple[int, int, int]]], None, None]:
        if not self._record_int:
            print("Warning: record_int is False, next_visual_generator will not work.")
            return
        while self.has_next_step():
            self.next_step()
            # 先把中间数据图像拷贝一份，顺便转换为 RGB 元组
            img_copy = [
                [hex_to_rgb(color) for color in row] for row in self._int_matrix
            ]
            # 把邻居的数据加入
            for nb in self._neighbors:
                img_copy[nb[0]][nb[1]] = hex_to_rgb(NEIGHBOR_COLOR)
            # 把被更新的邻居的数据加入
            for unb in self._updated_neighbors:
                img_copy[unb[0]][unb[1]] = hex_to_rgb(UPDATED_NEIGHBOR_COLOR)
            # 绘制目前的路径
            for pos in self.solved_path_coordinates:
                img_copy[pos[0]][pos[1]] = hex_to_rgb(PATH_COLOR)
            yield img_copy

    def solve(self):
        while self.has_next_step():
            self.next_step()

    @property
    def solved_path_coordinates(self) -> list[tuple[int, int]]:
        """
        最终的结果路径（坐标表示）

        :return: 路径上的结点坐标
        """
        return [node.pos for node in self._solution_path]

    @property
    def solved_path_cost(self) -> float:
        """
        最终结果路径的成本

        :return: 路径成本
        """
        return self._solution_path[-1].path_cost

    @property
    def state(self) -> AlgorithmState:
        return self._state
