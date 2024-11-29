from problems import generate_problem
from visualization import visualize_problem
from algorithms import AStarAlgorithm


if __name__ == "__main__":
    problem = generate_problem(150, 80, 0.7)
    visualize_problem(problem)
    algo = AStarAlgorithm(problem)
    # 执行算法
    algo.solve()
    print(algo.solved_path_cost)
    print(algo.solved_path_coordinates)
