# PathPlanningAlgorithms

课程作业。  

主要是探寻 A* 路径规划算法，咱附带实现了一下 JPS 版本的 A* 算法。

* PS：所有代码都有详细注释。

还别说，路径规划真挺有趣吧~ (。・∀・)ノ   

## 一些个人心得

### 梳理一下

Dijkstra 算法每次迭代中都会从目前**未确定结点中选择路径最短的结点**作为下一个落脚点，然后再扩展其所有的可行邻居结点。虽然 Dijkstra 能**保证找到最短路**，但是因为其扩展的速度特别慢，会均匀地探索各个路径，对于规模很大的图，求解速度实在是令人不敢恭维。

* 注意，Dijkstra **要求边权非负**。

A* 算法则是在 Dijkstra 的基础上，加入了启发式信息（这个信息就是“**是否沿着这个方向更有可能快速到达终点**”），使得其能够**优先扩展更有希望的结点**，从而**加速搜索过程**。

* A* 算法的性能因此也取决于启发式信息的计算方式（**不一定能找到最优路线**），在障碍物比较复杂时可能性能上会退化到 Dijkstra。  

Jump Point Search 优化的 A* 算法则要厉害多了，其通过一些策略大幅减少了**需要扩展的结点**数量，每次扩展的结点不一定是邻居，很可能是一个“跳点”。

----

💡 从上面也可以看出来，路径规划算法在性能方面一个很重要的点就是——**扩展结点的策略**。

* 扩展结点换个说法，其实就是程序认为**可能存在于最优路径上的结点**。程序会从这些结点出发寻找下一个可能的落脚点。
* 因此迭代过程中**需要扩展的结点越少**，需要检查的结点就越少，算法的内存消耗就越小，同时也能更快求解。

### JPS 算法碎碎念

JPS 优化的 A* 算法中最重要的两个概念：

1. 强迫邻居
2. 跳点

具体概念的介绍不多赘述，可以见参考文章。  

个人认为，要清晰地写代码实现 JPS-A* 算法，必须要搞清楚 JPS 的核心优化点在哪里——  
——没错，就是**对扩展结点策略**的优化！  

A* 算法每一次迭代在取出一个落脚点后，都会**扩展其所有邻居结点**；JPS 额外多了一些计算步骤，找到所谓的“跳点”，**扩展的是跳点而不是邻居**，而每次迭代中找到的跳点的数量要明显少很多，因此求解速度明显就要高很多。  

💡 简单来说，JPS 相比 A* 算法，改动的地方其实就是扩展结点的部分（比如在 A* 中就是寻找邻居并加入 `open_list` 这部分）。

## 容易被绕进去的地方

算法实现过程中最让我迷惑的一点：强制邻居（Forced Neighbor）到底是什么个地位？! 到底要不要把强制邻居加入 `open_list` ？很多文章都没讲清楚或没强调这一点。

折腾了好一段时间，总算把 JPS 算法实现后，我的答案是：  

* 不需要加入 `open_list`！ 

强制邻居的作用主要有两个：  

1. 用于判定跳点。（跳点周围肯定有强制邻居）
2. 用于**引导扩展结点**。

第 1 点很多文章都写的很明确，就不多赘述了。主要是第 2 点。

### 强制邻居是怎么引导扩展结点的？

我觉得整个过程中最让我迷糊的可能就是这个地方了。  

在从 `open_list` 取出一个结点后，在 A* 算法中有 8 个邻居，这其实也可以**看作是 8 个寻找方向**，但在 JPS 算法中，大部分情况下我们并不需要向所有 8 个方向进行搜索。

* 设结点 $x$ 的父结点是 $parent(x)$，所谓的“方向”就是 $parent(x)$ 指向 $x$ 的这个方向。

![8_directions](./assets/8_directions.png)  
> 图中标出来了 8 个“邻居”的位置，在 JPS 中这是 8 个寻找方向。  

* 上图中是一个起始节点，其没有父结点，因此需要向 8 个方向去查找跳点，将跳点加入 `open_list`。  

也就是从 `open_list` 取出的第二个结点开始，**就全都是跳点了**。

> 当然，把起始结点看作跳点也行。

从第二个结点开始就有父结点了，有父结点的结点就有方向。而有方向，JPS 就能顺着方向找跳点，有两种情况：  

1. **水平/竖直方向**。

    ![move_horizontally](./assets/move_horizontally.png) ![move_horizontally_with_obstacle](./assets/move_horizontally_with_obstacle.png)  

    沿着这个方向**一直走**直到遇到特定的（见参考文章）障碍物，就有了强制邻居（紫色），也就找到了跳点。

2. **对角方向**。

    ![move_diagonally](./assets/move_diagonally.png) ![move_diagonally_with_obstacle](./assets/move_diagonally_with_obstacle.png)  

    先沿着水平和竖直分量方向按 1 进行，然后再沿着对角线方向一直走直至遇到特定的（见参考文章）障碍物，就有了强制邻居（紫色），就找到了跳点。

> 特例：终点没有强制邻居。

注意，这里**强制邻居的作用**就来了！  

上面**每个右图**中绿色的结点肯定是从 `open_list` 中取出的跳点，因为其有强制邻居（紫色方块）：

1. 对于 1 中的情况，原本我们找跳点的方向是一直向右，**出现强制邻居后，新增了一个指向强制邻居的寻找方向**。  

2. 对于 2 中的情况，原本找跳点的方向是两个分量和一个对角方向，**出现强制邻居后，新增了一个强制邻居的寻找方向**。  

💡 因此每次取出跳点后的流程如下：

1. 计算这个跳点 $x$ 的方向（$parent(x)$ 指向 $x$ 的这个方向）。
2. 如果是水平/竖直方向，则有一个寻找方向；如果是对角方向，则有三个寻找方向（见参考文章）。  
3. 扫描一圈检查有几个强制邻居，每找到一个强制邻居，就加上一个寻找方向。  

    * 每个结点最多只有两个强制邻居，比如下面这张图：

        ![two obstacles](./assets/two_obstacles.png)  

        > 这里就有 $1+2=3$ 个找跳点的方向。  

4. 沿着这些方向去寻找跳点，并加入 `open_list`，再进入下一次迭代。  


所以说强制邻居的有引导跳点查找的功能，而 JPS 扩展的是跳点，因此强制邻居其实就是在引导结点的扩展。

## 目录结构

```bash
src
├── algorithms # 算法实现
│   ├── __init__.py
│   ├── a_star.py # A* 算法
│   ├── a_star_jps.py # A* 算法 Jump Point Search 版
│   ├── algorithm_base.py # 算法基类
│   ├── states.py # 算法状态枚举量
│   └── utils.py # 工具类，主要是方向 Direction 相关的类
├── exceptions
│   └── __init__.py # 一些自定义异常
├── main.py # 主类
├── problems
│   ├── __init__.py
│   ├── cell_status.py # 图中每个格子状态的枚举量
│   ├── drawer.py # 问题绘制模块
│   ├── generator.py # 随机问题生成
│   ├── problem.py # 问题类，用于表示每个二维栅格图
│   └── utils.py # 工具方法
├── test.py
├── tree.txt
└── visualization
    ├── __init__.py
    ├── algo_animator.py
    ├── problem_visualizer.py
    ├── result_visualizer.py
    └── utils.py
```

## 参考文献

* Harabor D, Grastien A. Online graph pruning for pathfinding on grid maps[C]//Proceedings of the AAAI conference on artificial intelligence. 2011, 25(1): 1114-1119.
* Duchoň F, Babinec A, Kajan M, et al. Path planning with modified a star algorithm for a mobile robot[J]. Procedia engineering, 2014, 96: 59-69.  
* Harabor D. Fast pathfinding via symmetry breaking[J]. Aigamedev Com, 2012.
* https://zerowidth.com/2013/a-visual-explanation-of-jump-point-search/  
* https://hakuya.me/algorithm/findpath/JPS%E7%AE%97%E6%B3%95%E5%88%86%E4%BA%AB/