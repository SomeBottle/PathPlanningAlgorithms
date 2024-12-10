# PathPlanningAlgorithms

课程作业。探寻 A* 路径规划算法。  

还别说，路径规划真挺有趣吧~ (。・∀・)ノ   

## 一些个人心得

### 梳理一下

Dijkstra 算法每次迭代中都会从目前**未确定结点中选择路径最短的结点**作为下一个落脚点，然后再扩展其所有的可行邻居结点。虽然 Dijkstra 能**保证找到最短路**，但是因为其扩展的速度特别慢，会均匀地探索各个路径，对于规模很大的图，求解速度实在是令人不敢恭维。

* 注意，Dijkstra **要求边权非负**。

A* 算法则是在 Dijkstra 的基础上，加入了启发式信息（这个信息就是“**是否沿着这个方向更有可能快速到达终点**”），使得其能够**优先扩展更有希望的结点**，从而**加速搜索过程**。

* A* 算法的性能因此也取决于启发式信息的计算方式，在障碍物比较复杂时可能性能上会退化到 Dijkstra。  

Jump Point Search 优化的 A* 算法则要厉害多了，其通过一些策略大幅减少了**需要扩展的结点**数量，每次扩展的结点不一定是邻居，很可能是一个“跳点”。

----

从上面也可以看出来，路径规划算法在性能方面一个很重要的点就是——**扩展结点的策略**。

* 扩展结点换个说法，其实就是程序认为**可能存在于最优路径上的结点**。程序会从这些结点出发寻找下一个可能的落脚点。
* 因此迭代过程中**需要扩展的结点越少**，需要检查的结点就越少，算法的内存消耗就越小，同时也能更快求解。

### JPS 算法碎碎念

JPS 优化的 A* 算法中最重要的两个概念：

1. 强迫邻居
2. 跳点

具体概念的介绍不多赘述。  

个人认为，要清晰地写代码实现 JPS-A* 算法，必须要搞清楚 JPS 的核心优化点在哪里——  
——没错，就是**对扩展结点策略**的优化！  

A* 算法每一次迭代在取出一个落脚点后，都会**扩展其所有邻居**；JPS 额外多了一些计算步骤，找到所谓的“跳点”，**扩展的是跳点而不是邻居**，而每次迭代中跳点的数量要明显少很多，因此求解速度明显就要高很多。  

💡 简单来说，JPS 相比 A* 算法，改动的地方其实就是扩展结点的部分（在 A* 中就是寻找邻居并加入 `open_list` 这部分）。

## 容易被绕进去的地方

最迷惑的一点：forced_neighbor 强制邻居到底是怎么处理的？! 到底要不要把强制邻居加入 OpenList ？很多文章都没讲清楚或没强调这一点。

很明确的是，强制邻居会影响到查询方向，如果在某个跳点有强制邻居，应当沿着邻居的方向继续展开搜索。


## 参考文献

* Harabor D, Grastien A. Online graph pruning for pathfinding on grid maps[C]//Proceedings of the AAAI conference on artificial intelligence. 2011, 25(1): 1114-1119.
* Duchoň F, Babinec A, Kajan M, et al. Path planning with modified a star algorithm for a mobile robot[J]. Procedia engineering, 2014, 96: 59-69.  
* Harabor D. Fast pathfinding via symmetry breaking[J]. Aigamedev Com, 2012.
* https://zerowidth.com/2013/a-visual-explanation-of-jump-point-search/  
* https://hakuya.me/algorithm/findpath/JPS%E7%AE%97%E6%B3%95%E5%88%86%E4%BA%AB/