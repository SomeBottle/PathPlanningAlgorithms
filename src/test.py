from problems import (
    generate_random_problem,
    generate_partial_ring_problem,
    draw_problem,
    Problem,
)
from visualization import (
    visualize_problem,
    visualize_result,
    AlgorithmAnimator,
)
from algorithms import AStarAlgorithm, AStarJPSDetourAlgorithmFixed

if __name__ == "__main__":
    """problem = generate_random_problem(150, 80, 0.7)
    visualize_problem(problem)
    algo = AStarAlgorithm(problem)
    # 执行算法
    algo.solve()
    visualize_result(problem, algo.solved_path_coordinates)"""

    """ problem = generate_partial_ring_problem(150, 80, ring_num=5, distance=30)
    visualize_problem(problem)
    algo = AStarAlgorithm(problem)
    # 执行算法
    algo.solve()
    visualize_result(problem, algo.solved_path_coordinates) """

    """ problem = draw_problem(150, 80)
    visualize_problem(problem)
    algo = AStarAlgorithm(problem)
    # 执行算法
    algo.solve()
    visualize_result(problem, algo.solved_path_coordinates) """

    """ problem = draw_problem(150, 80)

    visualize_problem(problem)

    algo = AStarAlgorithm(problem, record_int=True)

    ani = AlgorithmAnimator(algo)

    ani.show()

    print(algo.state) """

    problem = draw_problem(150, 80, close_diagonal_obstacles=False)

    visualize_problem(problem)

    problem = draw_problem(150, 80, close_diagonal_obstacles=True)

    visualize_problem(problem)

    """ problem = Problem.from_matrix(
        [
            [0, 0, 0, 1, 0, 0, 0, 4],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 3, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    ) """
    """ problem = Problem.from_matrix(
        [
            [0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 3, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ) """
    """ problem = Problem.from_matrix(
        [
            [0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 3, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ) """
    """ problem = Problem.from_matrix(
        [
            [0, 0, 0, 1, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0],
        ]
    ) """
    """ problem = Problem.from_matrix(
        [
            [0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ) """
    """ problem = Problem.from_matrix(
        [
            [0, 1, 4],
            [1, 3, 1],
            [0, 1, 0],
        ]
    )
 """
    # problem=Problem.from_file("./problem.pkl")

    algo = AStarJPSDetourAlgorithmFixed(problem, record_int=True,diagonal_obstacles=False)

    ani = AlgorithmAnimator(algo, interval=1)

    ani.show()

    print(algo.state)
    print(algo.solved_path_cost)

    algo = AStarAlgorithm(problem, record_int=True,diagonal_obstacles=False)

    ani = AlgorithmAnimator(algo, interval=1)

    ani.show()

    print(algo.state)
    print(algo.solved_path_cost)
