"""
A* 算法
"""

import heapq

from problems import Problem
from .states import AlgorithmState
from .algorithm_base import AlgorithmBase


# A* 算法的结点
class AStarNode:
    def __init__(
        self,
        path_cost=0.0,
        dist_to_end=0.0,
        parent=None,
        pos=(0, 0),
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

    def __init__(self, problem: Problem):
        super().__init__()
        # Open List 实际上是一个小根堆
        self._open_list = []
        # Open Dict 存储 (i,j) -> AStarNode 的映射
        self._open_dict = {}
        # Closed Dict 存储 (i,j) -> AStarNode 的映射
        self._closed_dict = {}
        self._problem = problem
        # 算法是否结束
        self._state: AlgorithmState = AlgorithmState.INITIALIZED
        # 记录最终的路径
        self._solution_path: list[AStarNode] = []
        # 把起点加入到开放列表中
        self._add_as_open(AStarNode(pos=problem.start))

    def _add_as_open(self, node: AStarNode):
        """
        将结点加入到开放列表中

        :param node: 要加入的结点
        """
        heapq.heappush(self._open_list, node)
        self._open_dict[node.pos] = node

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
        # 检查是不是终点
        if curr_node.pos == self._problem.end:
            # 到达终点
            self._state = AlgorithmState.SOLVED
            # 生成最终路径
            self._cache_solution(curr_node)
            return True

        # 寻找邻居，将可行的地方加入到 open_list
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4 个行进方向
        for deltas in directions:
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
        """
        获得算法的状态

        :return: 算法的状态
        """
        return self._state
