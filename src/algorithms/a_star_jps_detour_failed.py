"""
A* 算法 JPS 优化（带堆优化），尝试实现绕过对角障碍，但没有成功。  

这里我的绕路思路是像链表那样在中间插入绕路结点，可能是哪里写错了，没有成功。
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
        is_bypass=False,
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
        # 这个结点是不是一个绕路结点，用于处理绕过对角障碍物的情况，和 path_cost 有关
        self.is_bypass: bool = is_bypass

    def copy(self) -> "AStarNode":
        """
        完全拷贝这个结点，返回一个新的 AStarNode

        :return: 船新的 AStarNode
        """
        node_copy = AStarNode(
            path_cost=self.path_cost,
            dist_to_end=self.dist_to_end,
            parent=self.parent,
            pos=self.pos,
            is_bypass=self.is_bypass,
        )
        return node_copy

    def __lt__(self, other):
        """
        主要用于实现小根堆，heapq 应该是用 '<' 进行比较的

        :note: 算法每一次都要从 open_list 中找到距离起点和终点的距离之和最小的结点
        """
        return self.path_cost + self.dist_to_end < other.path_cost + other.dist_to_end


# A* 算法
class AStarJPSAlgorithm(AlgorithmBase):

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

    def _bypass_diagonal_obstacle(
        self, curr_node: AStarNode, direction: tuple[int, int]
    ) -> bool:
        """
        （注意，这里的实现和 A* 算法中的 _has_diagonal_obstacle 不同！！)

        检查从 curr_node 这个结点的位置开始沿着 direction 方向走会不会遇到对角障碍物，如果有，则尝试绕过。

        比如这些情况：

        ■             ■
        ↗ ■   ↗ ■   ↗

        * 注： JPS 只有第一种情况完全没法走，后面两种可以绕路
        * 只有对角方向移动时会遇到对角障碍物。
        * 绕路的实现方式是这样： parent -> curr_node 变为 parent -> (curr_node 副本) -> curr_node (坐标和开销变为绕路结点的)

            相当于把路线：

            ■
            ↗

            变成这样：

            ■ ↑
            → □  《 这个方框就是绕路结点，也就是修改后的 curr_node

            - 》 也就是说，同样是到达右上角，只不过我多经过一个绕路结点。
            - 》 这样相当于没有改变 curr_node 的内存指向，但实现了类似于链表中插入了新结点的效果。

        :param curr_node: 当前结点
        :param direction: 方向
        :return: 是否绕过对角障碍物（如果不是对角移动或者不考虑对角障碍物，直接返回 True）
        """
        # 不考虑对角障碍物 或 目前没有向对角方向走，就直接返回 False
        if not self._diagonal_obstacles or not Direction.is_diagonal(direction):
            return True
        # 如果对角上有障碍物，就不能在这个方向走了
        obs_1_coord = (curr_node.pos[0] + direction[0], curr_node.pos[1])
        obs_2_coord = (curr_node.pos[0], curr_node.pos[1] + direction[1])
        obs_1 = self._problem.is_obstacle(*obs_1_coord)
        obs_2 = self._problem.is_obstacle(*obs_2_coord)
        bypass_coord = None  # 绕路结点的坐标
        if obs_1 and obs_2:
            # 两个都堵上了，没法绕路
            return False
        elif obs_1:
            bypass_coord = obs_2_coord
        elif obs_2:
            bypass_coord = obs_1_coord

        if bypass_coord is not None:
            print(
                f"Bypassed: from {curr_node.pos} -> {(curr_node.pos[0] + direction[0], curr_node.pos[1]+direction[1])} to {curr_node.pos} -> {bypass_coord}"
            )
            # 需要进行绕路，先拷贝当前结点
            curr_node_copy = curr_node.copy()
            # 然后把 curr_node 改造成绕路结点
            curr_node.pos = bypass_coord
            # 绕路肯定是在水平或竖直方向上走的，所以开销加 1
            curr_node.path_cost = curr_node_copy.path_cost + 1
            curr_node.dist_to_end = self._problem.dist_to_end(*bypass_coord)
            curr_node.parent = curr_node_copy
            # 标记发生了绕路，这样后面在计算这个路径的长度时是 +1 而不是 + sqrt(2)
            curr_node.is_bypass = True

        return True

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

    def _find_directions(self, curr_node: AStarNode) -> list[tuple[int, int]]:
        """
        找到 curr_node 应该行进的方向 (最差情况有 8 个方向)

        :param curr_node: 当前结点
        :return: 可行的方向列表
        """
        possible_directions = []
        if curr_node.parent is None:
            # 如果没有父节点，则八个方向都需要考虑
            possible_directions = [d for d in DIRECTIONS]
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
            if (
                not self._problem.in_bounds(*next_pos)
                or self._problem.is_blocked(*next_pos)
                or not self._bypass_diagonal_obstacle(curr_node, d)
            ):
                # 如果有障碍物，这个方向就不可行
                continue
            if next_pos in self._closed_dict:
                # 如果这个位置已经访问过并确定下来了，也跳过
                continue

            res.append(d)

        return res

    def _jump(
        self, curr_node: AStarNode, neighbor_direction: tuple[int, int]
    ) -> AStarNode | None:
        """
        从 neighbor_direction 指向的邻居开始，沿着这个方向找到跳点

        :param curr_node: 当前结点
        :param neighbor_direction: 邻居方向
        :return: 跳点结点，没找到就是 None
        """
        di, dj = neighbor_direction
        # 前进过程中每一步的长度
        step_len = math.sqrt(di**2 + dj**2)
        # 先计算 neighbor_direction 指向的邻居坐标
        if curr_node.is_bypass:
            # 当前结点如果是绕路结点，则应该用父节点的坐标计算实际的邻居坐标
            i, j = Direction.step(curr_node.parent.pos, neighbor_direction)
            # 当前结点因为绕路了，走到 curr_node.parent.pos + neighbor_direction 这个位置实际上走了 2 步
            dist = 0 + 2
        else:
            i, j = Direction.step(curr_node.pos, neighbor_direction)
            # 目前距离 curr_node 的长度，因为现在已经移动到邻居了，距离加一步
            dist = 0 + step_len
        # 是否在按对角方向行进
        diagonal = Direction.is_diagonal(neighbor_direction)

        # 存储前进搜索过程中的父结点，因为有绕路的情况，父节点可能会突变
        curr_parent = curr_node

        # 按照这个方向向前走
        while True:
            if not self._problem.in_bounds(i, j) or self._problem.is_obstacle(i, j):
                # 1. 走到边界外或者迎头撞上障碍物了
                return None
            print(f"Walk to {(i,j)}")
            # 前进搜索时利用的临时结点
            temp_node = AStarNode(
                path_cost=curr_node.path_cost + dist,
                parent=curr_parent,
                pos=(i, j),
                dist_to_end=self._problem.dist_to_end(i, j),
            )
            if not self._bypass_diagonal_obstacle(temp_node, neighbor_direction):
                # 1. 遇到了对角线障碍物，且没能绕开
                print(
                    f"obstacle not passable, try to move from {temp_node.pos} to {(temp_node.pos[0]+neighbor_direction[0],temp_node.pos[1]+neighbor_direction[1])}"
                )
                return None
            # 2. 如果正好遇到了最终结点，直接返回这个结点作为跳点
            if (i, j) == self._problem.end:
                return temp_node
            # 3. 如果是对角线方向，先要向两个分量方向寻找跳点
            if diagonal:
                print("FINDING COMPONENT")
                if (
                    self._jump(temp_node, (di, 0)) is not None
                    or self._jump(temp_node, (0, dj)) is not None
                ):
                    # 如果找到了跳点，当前结点就是间接跳点
                    print("\t act as indirect jump point")
                    print("END FINDING COMPONENT")
                    return temp_node
                print("END FINDING COMPONENT")
            # 4. 判断当前结点是否有强制邻居需要考虑
            if len(self._get_forced_neighbors((i, j), neighbor_direction)) > 0:
                # 当前结点是直接跳点
                print("\t act as direct jump point")
                return temp_node
            # 5. 上面条件都没满足，继续按照这个方向走
            # 论文中这里写成递归了，实际上没必要。
            i += di
            j += dj
            if temp_node.is_bypass:
                # 如果绕路了，接下来实际只前行了 1
                dist += 2
                # 为了最终获取路径时能考虑到绕路点，需要在这里突变父结点
                curr_parent = temp_node
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

        这里因为是 JPS，存储的是跳点，需要进行一定的填充

        :return: 路径上的结点坐标
        """
        padded_solution = []
        for i, node in enumerate(self._solution_path):
            if i > 0:
                di, dj = Direction.get(self._solution_path[i - 1].pos, node.pos)
                p_i, p_j = self._solution_path[i - 1].pos
                while True:
                    p_i += di
                    p_j += dj
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
