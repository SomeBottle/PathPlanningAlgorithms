"""
A* 算法 JPS 优化（带堆优化），且支持绕过对角障碍，路径搜索有修正。  

这里绕过对角障碍的思路是，遇到障碍物时如果能绕过：

1. 先记录绕路结点，并不是实际将其加入结点链表。
2. 立马修正当前的路径长度 path_cost，因为绕路肯定会使得路径变长，会影响算法的搜索过程，必须立即更新。

然后在生成路径的时候，把记录的绕路结点都加上即可。  
-----

除此之外本算法还修正了绕路后的搜索方向，保证能和 A* 算法得到一致的路径代价：

3. 把绕路结点**临时加入**开放列表，并指定其移动方向，以让算法考虑可能的新关键路径。

    - SomeBottle 20241213
"""

import heapq
import math

from problems import Problem
from .states import AlgorithmState
from .algorithm_base import AlgorithmBase
from .utils import Direction

# 这里如果从 visualization 导入会造成环型导入
from visualization.utils import hex_to_rgb
from typing import Generator

# 八个方向
DIRECTIONS = Direction.all()

# 用于展示中间结果
CLOSED_COLOR = "#A59D84"  # 确定下来的路径的颜色
OPEN_COLOR = "#FFD2A0"  # 待探索的路径的颜色
NEIGHBOR_COLOR = "#85A98F"  # 每个位置邻居的颜色
UPDATED_NEIGHBOR_COLOR = "#D91656"  # 更新了的邻居的颜色
PATH_COLOR = "#D91656"  # 走过的路径的颜色


# A* 算法的结点
class AStarNode:
    def __init__(
        self,
        path_cost=0.0,
        dist_to_end=0.0,
        parent=None,
        pos=(0, 0),
        forced_direction=None,
    ):
        # 这个结点距离起点的路径的代价 g(x)
        self.path_cost: float = path_cost
        # 这个结点距离终点的预估代价 h(x)
        self.dist_to_end: float = dist_to_end
        # 这个结点的父节点，用于找到路径
        self.parent: AStarNode | None = parent
        # 这个结点的位置
        self.pos: tuple[int, int] = pos
        # 强制移动方向，如果指定了这个方向，不会用 parent 和当前结点的坐标来计算方向
        self.forced_direction = forced_direction

    def __lt__(self, other):
        """
        主要用于实现小根堆，heapq 应该是用 '<' 进行比较的

        :note: 算法每一次都要从 open_list 中找到距离起点和终点的距离之和最小的结点
        """
        return self.path_cost + self.dist_to_end < other.path_cost + other.dist_to_end


# A* 算法
class AStarJPSDetourAlgorithmFixed(AlgorithmBase):

    def __init__(self, problem: Problem, record_int=False, diagonal_obstacles=True):
        # Open List 实际上是一个小根堆
        self._open_list = []
        # Open Dict 存储 (i,j) -> AStarNode 的映射
        self._open_dict = {}
        # Closed Dict 存储 (i,j) -> AStarNode 的映射
        self._closed_dict = {}
        self._problem = problem
        # 记录最终的路径
        self._solution_path: list[AStarNode] = []
        # 是否考虑对角障碍物
        self._diagonal_obstacles = diagonal_obstacles
        # 存储绕路结点坐标
        self._bypass_nodes = {}
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

    def _add_bypass_node(
        self,
        bypass_pos: tuple[int, int],
        pos_1: tuple[int, int],
        pos_2: tuple[int, int],
        parent_node: AStarNode,
    ):
        """
        添加绕路结点，主要做以下两件事：

        1. 存储绕路结点坐标，标记从 pos_1 坐标到 pos_2 坐标需要经过一个绕路结点 bypass
        2. 把绕路结点**临时**加入到开放列表中，指定搜索方向为 pos_1 -> pos_2 的方向，以修正算法的搜索策略

        :param bypass_pos: 绕路结点坐标
        :param pos_1: 前一个坐标
        :param pos_2: 后一个坐标
        :param parent_node: 绕路结点的父结点
        """
        if self._bypass_nodes.get(pos_1) is None:
            self._bypass_nodes[pos_1] = {}
        self._bypass_nodes[pos_1][pos_2] = bypass_pos

        # 把绕路结点加入开放列表

        bypass_node = AStarNode(
            dist_to_end=self._problem.dist_to_end(*bypass_pos),
            pos=bypass_pos,
            # 强制这个绕路结点处的搜索方向沿着和 pos_1 -> pos_2 平行的方向
            forced_direction=Direction.get(pos_1, pos_2),
        )
        real_parent_node = parent_node
        if parent_node.pos != pos_1:
            # 父结点不是前一个结点，否则还要在 pos_1 处额外建立一个父结点
            real_parent_node = AStarNode(
                path_cost=parent_node.path_cost
                + Direction.dist(parent_node.pos, pos_1),
                dist_to_end=self._problem.dist_to_end(*pos_1),
                pos=pos_1,
                parent=parent_node,
            )

        bypass_node.parent = real_parent_node
        bypass_node.path_cost = real_parent_node.path_cost + 1
        self._add_as_open(bypass_node, bypass_node=True)

    def _get_bypass_pos(
        self,
        pos_1: tuple[int, int],
        pos_2: tuple[int, int],
    ) -> tuple[int, int] | None:
        """
        根据 pos_1 和 pos_2 取出其要绕路的结点坐标，可能没有

        :param pos_1: 前一个结点的坐标
        :param pos_2: 后一个结点的坐标
        :return: 绕路结点坐标，没有的话会返回 None
        """
        if self._bypass_nodes.get(pos_1) is None:
            return None
        return self._bypass_nodes[pos_1].get(pos_2)

    def _add_as_open(self, node: AStarNode, bypass_node: bool = False):
        """
        将结点加入到开放列表中

        :param node: 要加入的结点
        :param bypass_node: 这个结点是不是绕路结点，绕路结点只会临时加入堆中。
        """
        heapq.heappush(self._open_list, node)
        # 绕路结点只临时加入堆，并不作为实际的绕路结点处理
        if not bypass_node:
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
        # 如果是临时添加的绕路结点(forced_direction 不为 None)，本身就没有加入 _open_dict，不作处理。
        if node.forced_direction is None:
            del self._open_dict[node.pos]
        return node

    def _cache_solution(self, end_node: AStarNode):
        """
        从最后一个结点，通过父结点开始逆推，得到最终的结果路径

        注意，这个方法只是把 AStarNode 按顺序放在一个列表中，没有加入绕路结点。

        :param end_node: 最后一个结点
        """
        self._solution_path = []
        while end_node is not None:
            self._solution_path.append(end_node)
            end_node = end_node.parent
        self._solution_path.reverse()

    def _get_diagonal_obstacles(
        self, curr_pos: tuple[int, int], direction: tuple[int, int]
    ) -> tuple[bool, tuple[int, int] | None]:
        """
        检查 curr_pos 这个地方沿着 direction 方向走是否遇到对角障碍物

        比如这些情况：

        ■             ■
        ↗ ■   ↗ ■   ↗

        * 只有对角方向移动时会遇到对角障碍物。

        （如果 diagonal_obstacles=False 会直接返回 (False, None)）

        :param curr_pos: 当前位置
        :param direction: 方向
        :return: (是否有对角障碍物, 绕路结点坐标)，绕路节点坐标可能是 None
        """
        # 不考虑对角障碍物 或 目前没有向对角方向走，就直接返回 False
        if not self._diagonal_obstacles or not Direction.is_diagonal(direction):
            return (False, None)
        obs1_coord = curr_pos[0] + direction[0], curr_pos[1]
        obs2_coord = curr_pos[0], curr_pos[1] + direction[1]
        # 这里用 is_blocked，越界的地方也要算进去
        obs1 = self._problem.is_blocked(*obs1_coord)
        obs2 = self._problem.is_blocked(*obs2_coord)
        if obs1 and obs2:
            # 此路不通
            bypass_pos = None
        elif obs1:
            bypass_pos = obs2_coord
        elif obs2:
            bypass_pos = obs1_coord
        else:
            # 没有堵塞
            return (False, None)

        # 检查穿过对角障碍后是否越界，因为 is_blocked 对于越界的情况也会返回 True，如果穿过障碍物就越界了，就根本没必要绕路了，反正是死路一条
        if not self._problem.in_bounds(*Direction.step(curr_pos, direction)):
            bypass_pos = None

        return (True, bypass_pos)

    def _get_forced_neighbors(
        self, coordinate: tuple[int, int], direction: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        找到沿着 direction 方向走到 coordinate 处，旁边是否有强制邻居，如果有则返回到达这些强制邻居的坐标

        :param coordinate: 当前坐标
        :param direction: 当前方向
        :return: 前往强制邻居的方向 (di,dj) 列表, di,dj ∈ {-1,0,1}
        """
        i, j = coordinate
        di, dj = direction
        forced_neighbors = []
        if Direction.is_diagonal(direction):
            # 对角线方向运动。比如向右上方运动，左方或下方若有障碍物，则有强制邻居
            check_coord_1 = (i - di, j)
            check_coord_2 = (i, j - dj)
            if self._problem.is_obstacle(*check_coord_1):
                forced_neighbors.append((-di, dj))
            if self._problem.is_obstacle(*check_coord_2):
                forced_neighbors.append((di, -dj))
        else:
            # 水平或者竖直方向运动。比如向右方运动，上方或下方若有障碍物，则有强制邻居
            # 获得正交方向
            for d in Direction.orthogonal(direction):
                odi, odj = d  # 正交方向
                if self._problem.is_obstacle(i + odi, j + odj):
                    forced_neighbors.append((odi + di, odj + dj))

        return forced_neighbors

    def _find_directions(self, curr_node: AStarNode) -> list[tuple[int, int, int]]:
        """
        找到 curr_node 应该行进的方向 (最差情况有 8 个方向)

        :param curr_node: 当前结点
        :return: 可行的方向列表 (di, dj, 沿着这方向走一步的长度)
        """
        possible_directions = []
        if curr_node.parent is None:
            # 如果没有父节点，则八个方向都需要考虑
            possible_directions = [d for d in DIRECTIONS]
        else:
            if curr_node.forced_direction is not None:
                # 这个结点被强制指定了移动方向
                curr_dir = curr_node.forced_direction
            else:
                # 有父节点，计算方向
                curr_dir = Direction.get(curr_node.parent.pos, curr_node.pos)

            diagonal = Direction.is_diagonal(curr_dir)  # 是否是对角线方向运动
            di, dj = curr_dir
            # 首先后一个位置肯定是要考虑的，这是一个自然邻居
            possible_directions.append(curr_dir)
            # 如果是对角方向走，还有两个自然邻居
            if diagonal:
                possible_directions.append((di, 0))
                possible_directions.append((0, dj))
            # 到这里，这个跳点必然有强制邻居
            # 找到强制邻居的方向
            forced_neighbors_directions = self._get_forced_neighbors(
                curr_node.pos, curr_dir
            )
            # 有强制邻居的话也要把强制邻居的方向考虑在内
            possible_directions.extend(forced_neighbors_directions)

        # 检查这些方向是不是都是可行的
        res = []
        for d in possible_directions:
            # 按照 d 方向走一步后的坐标
            next_pos = Direction.step(curr_node.pos, d)
            diagonal_ob, bypass_coord = self._get_diagonal_obstacles(curr_node.pos, d)
            if (
                not self._problem.in_bounds(*next_pos)
                or self._problem.is_blocked(*next_pos)
                or (diagonal_ob and bypass_coord is None)
            ):
                # 如果有障碍物，这个方向就不可行
                continue
            if next_pos in self._closed_dict:
                # 如果这个位置已经访问过并确定下来了，也跳过
                continue

            # 沿着这个方向走一步的长度
            if bypass_coord is not None:
                # 只要有绕路结点，原本斜着走要拆分为走 2 步
                first_step_len = 2
                # 另外记录绕路结点
                self._add_bypass_node(bypass_coord, curr_node.pos, next_pos, curr_node)
            else:
                first_step_len = math.sqrt(d[0] ** 2 + d[1] ** 2)

            res.append((d[0], d[1], first_step_len))

        return res

    def _jump(
        self, curr_node: AStarNode, search_direction: tuple[int, int, int]
    ) -> AStarNode | None:
        """
        从 neighbor_direction 指向的邻居开始，沿着这个方向找到跳点

        :param curr_node: 当前结点
        :param neighbor_direction: 搜索方向 (di, dj, 沿着 search_direction 走第一步的长度)
        :return: 跳点结点，没找到就是 None
        """
        di_dj = search_direction[:2]
        # 先计算 neighbor_direction 指向的邻居坐标
        i, j = Direction.step(curr_node.pos, di_dj)
        di, dj, first_step_len = search_direction
        # 是否在按对角方向行进
        diagonal = Direction.is_diagonal(di_dj)

        # 除了第一步外每一步的长度
        step_len = math.sqrt(di**2 + dj**2)

        # 目前距离 curr_node 的长度，因为现在已经移动到邻居了，距离加一步
        dist = 0 + first_step_len
        # 按照这个方向向前走
        while True:
            if not self._problem.in_bounds(i, j) or self._problem.is_obstacle(i, j):
                # 1. 走到边界外或者迎头撞上障碍物了
                return None
            # 行进过程中的临时结点
            tmp_node = AStarNode(
                path_cost=curr_node.path_cost + dist,
                parent=curr_node,
                pos=(i, j),
                dist_to_end=self._problem.dist_to_end(i, j),
            )
            diagonal_ob, bypass_coord = self._get_diagonal_obstacles(
                tmp_node.pos, di_dj
            )

            # 2. 如果正好遇到了最终结点，直接返回这个结点作为跳点
            # 注意这个要放在对角障碍物判断的前面，否则终点在角落里时 diagonal_ob=True，本方法会返回，导致终点被忽略。
            if (i, j) == self._problem.end:
                return tmp_node

            # 3. 如果是对角线方向，先要向两个分量方向寻找跳点
            if diagonal:
                if (
                    self._jump(tmp_node, (di, 0, 1)) is not None
                    or self._jump(tmp_node, (0, dj, 1)) is not None
                ):
                    # 如果找到了跳点，当前结点就是间接跳点
                    return tmp_node

            # 4. 判断当前结点是否有强制邻居需要考虑
            if len(self._get_forced_neighbors((i, j), di_dj)) > 0:
                # 当前结点是直接跳点
                return tmp_node

            # 5. 如果被对角障碍物堵塞了，没法继续前行了
            if diagonal_ob and bypass_coord is None:
                return None

            # 6. 上面条件都没满足，继续按照这个方向走
            # 论文中这里写成递归了，实际上没必要。

            i += di
            j += dj

            # 有绕路情况一定要及时修正 path_cost，不然会影响算法的搜索决策
            if bypass_coord is not None:
                # 如果需要绕路，这一步的步长肯定是 2
                dist += 2
                # 记录绕路结点
                self._add_bypass_node(bypass_coord, tmp_node.pos, (i, j), curr_node)
            else:
                dist += step_len

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
        else:
            self._state = AlgorithmState.RUNNING

        # 取得到起点和终点距离之和最小的结点
        curr_node: AStarNode = self._pop_min_open()
        # 绕路结点是临时添加的，不加入 closed_dict
        if curr_node.forced_direction is None:
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

        # --------------------------- Jump Point Search 核心部分
        for direction in self._find_directions(curr_node):
            # direction 是一个可行的跳点寻找方向
            jump_node = self._jump(curr_node, direction)

            if jump_node is not None:
                # =========== 更新中间数据 ===========
                if self._record_int:
                    self._neighbors.append(jump_node.pos)  # 记录邻居
                # =========== 中间数据更新完成 ===========
                # 找到了跳点，先检查有没有加入过 open_list
                if jump_node.pos in self._open_dict:
                    # 如果加入过，看看能不能更新路径
                    exist_node: AStarNode = self._open_dict[jump_node.pos]
                    if jump_node.path_cost < exist_node.path_cost:
                        # 如果新的路径代价更小，更新路径
                        exist_node.parent = curr_node
                        exist_node.path_cost = jump_node.path_cost
                        # 更新小根堆
                        heapq.heapify(self._open_list)
                        # =========== 更新中间数据 ===========
                        if self._record_int:
                            self._updated_neighbors.append(jump_node.pos)
                        # =========== 中间数据更新完成 ===========
                else:
                    # 否则直接加入 open_list
                    self._add_as_open(jump_node)

        # --------------------------- Jump Point Search 核心部分结束

        return True

    def next_visual_generator(
        self,
    ) -> Generator[list[list[tuple[int, int, int]]], None, None]:
        if not self._record_int:
            print(
                "Warning: record_int of the algorithm is False, next_visual_generator will not work."
            )
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

        1. 这里因为是 JPS，存储的是跳点，需要进行一定的填充。
        2. 途中可能有一些绕路结点，在这里要加到路径上。

        :return: 路径上的结点坐标
        """
        padded_solution = []
        for i, node in enumerate(self._solution_path):
            if i > 0:
                di, dj = Direction.get(self._solution_path[i - 1].pos, node.pos)
                p_i, p_j = self._solution_path[i - 1].pos
                while True:
                    next_i = p_i + di
                    next_j = p_j + dj
                    bypass_pos = self._get_bypass_pos((p_i, p_j), (next_i, next_j))
                    if bypass_pos is not None:
                        # 如果中间有绕路结点就加上
                        padded_solution.append(bypass_pos)
                    p_i = next_i
                    p_j = next_j
                    if (p_i, p_j) == node.pos:
                        break
                    padded_solution.append((p_i, p_j))
            padded_solution.append(node.pos)

        return padded_solution

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
