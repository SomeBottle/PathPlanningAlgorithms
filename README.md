# PathPlanningAlgorithms

课程作业。探寻 A* 路径规划算法。  

还别说，路径规划真挺有趣吧~ (。・∀・)ノ   

## 关于 JPS 优化的一些个人心得

JPS 即 Jump Point Search，应该是很常见也很经典的 A* 优化算法了，虽然看上去思想并不难，但动手写代码时还是容易把自己绕进去。  

JPS 优化的 A* 算法中最重要的两个概念：

1. 强迫邻居
2. 跳点


主要需要注意以下几点：

1. 每次从 OpenList 取出一个结点的时候，如果这个结点**有父节点**（如果是起始结点，是没有父节点的），那么 JPS 第一步注重的区域就是**以这个结点为中心的九个格子**，结点的行进方向可以从其和父节点的坐标差得出。

最迷惑的一点：forced_neighbor 强制邻居到底是怎么处理的？! 到底要不要把强制邻居加入 OpenList ？很多文章都没讲清楚或没强调这一点。

很明确的是，强制邻居会影响到查询方向，如果在某个跳点有强制邻居，应当沿着邻居的方向继续展开搜索。


## 参考文献

* Harabor D, Grastien A. Online graph pruning for pathfinding on grid maps[C]//Proceedings of the AAAI conference on artificial intelligence. 2011, 25(1): 1114-1119.
* Duchoň F, Babinec A, Kajan M, et al. Path planning with modified a star algorithm for a mobile robot[J]. Procedia engineering, 2014, 96: 59-69.  
* Harabor D. Fast pathfinding via symmetry breaking[J]. Aigamedev Com, 2012.
* https://zerowidth.com/2013/a-visual-explanation-of-jump-point-search/  
* https://hakuya.me/algorithm/findpath/JPS%E7%AE%97%E6%B3%95%E5%88%86%E4%BA%AB/