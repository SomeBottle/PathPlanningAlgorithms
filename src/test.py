from problems import generate_problem
from visualization import visualize_problem, visualize_result
from algorithms import AStarAlgorithm


if __name__ == "__main__":
    problem = generate_problem(300, 200, 0.5)
    visualize_problem(problem)
    algo = AStarAlgorithm(problem)
    # 执行算法
    algo.solve()
    print(algo.solved_path_cost)
    print(algo.solved_path_coordinates)
    visualize_result(problem, algo.solved_path_coordinates)
