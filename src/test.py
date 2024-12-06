from problems import (
    generate_random_problem,
    generate_partial_ring_problem,
    draw_problem,
)
from visualization import (
    visualize_problem,
    visualize_result,
    AlgorithmAnimator,
)
from algorithms import AStarAlgorithm


if __name__ == "__main__":
    """problem = generate_random_problem(150, 80, 0.7)
    visualize_problem(problem)
    algo = AStarAlgorithm(problem)
    # 执行算法
    algo.solve()
    print(algo.solved_path_cost)
    print(algo.solved_path_coordinates)
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

    problem = draw_problem(150, 80)

    visualize_problem(problem)

    algo = AStarAlgorithm(problem, record_int=True)

    ani = AlgorithmAnimator(algo)

    ani.save('animation.mp4')

    print(algo.state)
