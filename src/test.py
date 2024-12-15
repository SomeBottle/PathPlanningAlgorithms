import time
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
from algorithms import (
    AStarAlgorithm,
    AStarJPSAlgorithm,
    AStarJPSDetourAlgorithm,
    AStarJPSDetourAlgorithmFixed,
)

PROBLEM_MATRIX = [
    [0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 3, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
]

PROBLEM_FILE = "./problem.bin"


def show_animation(ani: AlgorithmAnimator):
    """
    展示动画，灵活判断是不是在 IPython 环境下

    :param ani: AlgorithmAnimator 对象
    """
    try:
        get_ipython()
        from IPython.display import Video, display

        video_name = f"temp_output_{int(time.time())}.mp4"
        print(
            "IPython environment detected, rendering video... It will be showed later."
        )
        ani.save(video_name, fps=12)
        display(Video(video_name, embed=True))

    except NameError as e:
        # 不在 IPython 环境下
        ani.show()


def test_load_and_visualize_problem():
    """
    从文件中载入问题并可视化展示
    """
    # 注：./problem.bin 采用 pickle v4 序列化，要求 Python 版本 >= 3.4
    problem = Problem.from_file(PROBLEM_FILE)
    visualize_problem(problem)


def test_partial_ring_problem():
    """
    测试生成带随机半环状障碍物的问题

    （老实说，感觉我的实现有问题 ㄟ( ▔, ▔ )ㄏ ）
    """
    problem = generate_partial_ring_problem(150, 80, ring_num=5, distance=30)
    visualize_problem(problem)


def test_random_problem():
    """
    测试有大量随机障碍物的问题的生成
    """
    problem = generate_random_problem(150, 80, 0.7)
    visualize_problem(problem)


def test_draw_and_save_problem():
    """
    绘制问题并持久化存储为 ./problem_{秒级时间戳}.bin
    """
    problem = draw_problem(150, 80)
    problem.save(f"./problem_{int(time.time())}.bin")


def test_load_problem_from_matrix():
    """
    尝试从矩阵中载入问题
    """
    problem = Problem.from_matrix(PROBLEM_MATRIX)
    visualize_problem(problem)


def test_close_diagonal_obstacles():
    """
    测试消除对角障碍物
    """
    problem = Problem.from_matrix(PROBLEM_MATRIX)
    visualize_problem(problem)
    problem = Problem.from_matrix(PROBLEM_MATRIX, close_diagonal_obstacles=True)
    visualize_problem(problem)


def test_animate_jps_detour_fixed():
    """
    尝试执行支持绕路的 JPS 算法，动画展示
    """
    problem = Problem.from_file(PROBLEM_FILE)
    visualize_problem(problem)
    # 要动画展示算法过程，record_int 必须设定为 True，以记录算法执行的中间过程
    algo = AStarJPSDetourAlgorithmFixed(problem, record_int=True)
    ani = AlgorithmAnimator(algo, interval=1)
    show_animation(ani)
    print(f"算法求解状态: {algo.state}")
    print(f"求解得到的路径长度：{algo.solved_path_cost}")


def test_animate_ignore_diagonal_obstacles():
    """
    忽略对角障碍，执行 JPS 算法，动画展示
    """
    problem = Problem.from_file(PROBLEM_FILE)
    visualize_problem(problem)
    # 要动画展示算法过程，record_int 必须设定为 True，以记录算法执行的中间过程
    algo = AStarJPSDetourAlgorithmFixed(
        problem, record_int=True, diagonal_obstacles=False
    )
    ani = AlgorithmAnimator(algo, interval=1)
    show_animation(ani)
    print(f"算法求解状态: {algo.state}")
    print(f"求解得到的路径长度：{algo.solved_path_cost}")


def test_animate_a_star():
    """
    执行 A* 算法，动画展示
    """
    problem = Problem.from_file(PROBLEM_FILE)
    visualize_problem(problem)
    # 要动画展示算法过程，record_int 必须设定为 True，以记录算法执行的中间过程
    algo = AStarAlgorithm(problem, record_int=True)
    ani = AlgorithmAnimator(algo, interval=1)
    show_animation(ani)
    print(f"算法求解状态: {algo.state}")
    print(f"求解得到的路径长度：{algo.solved_path_cost}")


def test_animation_save():
    """
    测试动画视频的渲染和存储（需要有 ffmpeg）
    """
    problem = Problem.from_file(PROBLEM_FILE)
    visualize_problem(problem)
    # 要动画展示算法过程，record_int 必须设定为 True，以记录算法执行的中间过程
    algo = AStarJPSDetourAlgorithmFixed(problem, record_int=True)
    ani = AlgorithmAnimator(algo, interval=1)
    ani.save("./animation.mp4", fps=12)


def test_both_a_star_and_jps(diagonal_obstacles=True):
    """
    同时用 A* 和支持绕路的 JPS 求解问题，比较

    :param diagonal_obstacles: 是否考虑对角障碍
    """
    problem = Problem.from_file(PROBLEM_FILE)
    # problem = generate_random_problem(150, 80, 0.7)
    # A*
    algo = AStarAlgorithm(problem, diagonal_obstacles=diagonal_obstacles)
    start_time = time.perf_counter_ns()
    # 调用 solve 直接进行求解
    algo.solve()
    a_star_time_consumed = time.perf_counter_ns() - start_time
    a_star_path_cost = algo.solved_path_cost

    # JPS Detour
    algo = AStarJPSDetourAlgorithm(problem, diagonal_obstacles=diagonal_obstacles)
    start_time = time.perf_counter_ns()
    algo.solve()
    jps_time_consumed = time.perf_counter_ns() - start_time
    jps_path_cost = algo.solved_path_cost

    # JPS Detour Fixed
    algo = AStarJPSDetourAlgorithmFixed(problem, diagonal_obstacles=diagonal_obstacles)
    start_time = time.perf_counter_ns()
    algo.solve()
    jps_fixed_time_consumed = time.perf_counter_ns() - start_time
    jps_fixed_path_cost = algo.solved_path_cost

    print(f"A* 算法求解耗时：{a_star_time_consumed/1e6:.3f} ms")
    print(f"A* 算法求解路径长度：{a_star_path_cost:.3f}")
    print(f"JPS（对角障碍绕路）算法求解耗时：{jps_time_consumed/1e6:.3f} ms")
    print(f"JPS （对角障碍绕路）算法求解路径长度：{jps_path_cost:.3f}")
    print(
        f"JPS（对角障碍绕路，修正）算法求解耗时：{jps_fixed_time_consumed/1e6:.3f} ms"
    )
    print(f"JPS （对角障碍绕路，修正）算法求解路径长度：{jps_fixed_path_cost:.3f}")


def test_visualize_result():
    """
    测试可视化算法求解结果

    这里采用了 AStarJPSAlgorithm 算法，它不支持绕路。
    """
    problem = Problem.from_file(PROBLEM_FILE)
    visualize_problem(problem)
    algo = AStarJPSAlgorithm(problem)
    algo.solve()
    # 可视化算法求解得到的路线
    visualize_result(problem, algo.solved_path_coordinates)


if __name__ == "__main__":
    pass
