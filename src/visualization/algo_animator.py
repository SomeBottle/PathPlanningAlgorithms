"""
动画可视化算法过程的模块
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from algorithms import AlgorithmBase
from .problem_visualizer import visualize_as_color_matrix
from .utils import scaled_figsize, hex_to_rgb


class AlgorithmAnimator:
    def __init__(self, algo: AlgorithmBase):
        """
        初始化算法动画可视化实例.

        * 注：初始化算法时请指定 record_int=True，不然算法不会记录中间结果。

        :param algo: 算法实例。本方法会利用其 next_visual_generator 生成器生成动画
        """
        self._algo = algo

    def _render(self, save_to_file: str | None = None):
        """
        渲染动画

        :param save_to_file: 保存动画的文件路径，如果为 None 则不保存
        """
        generator = self._algo.next_visual_generator()

        w = self._algo.problem.width
        h = self._algo.problem.height
        # 计算宽高比例确定图表大小
        w, h = scaled_figsize(w, h)
        fig, ax = plt.subplots(figsize=(w, h))
        ax.axis("off")
        img = ax.imshow(next(generator), aspect="auto")

        def _update(algo_frame):
            # 先绘制问题图像，然后再把算法生成的图像覆盖上去
            problem_img = visualize_as_color_matrix(self._algo.problem)
            empty_color = hex_to_rgb(self._algo.EMPTY_COLOR)  # img_frame 空白处的颜色
            update_img = [
                [
                    (
                        problem_img[i][j]
                        if algo_frame[i][j] == empty_color
                        else algo_frame[i][j]
                    )
                    for j in range(len(problem_img[i]))
                ]
                for i in range(len(problem_img))
            ]
            img.set_data(update_img)
            return (img,)

        # 创建动画
        ani = animation.FuncAnimation(
            fig, _update, interval=1, frames=generator, blit=True, repeat=False
        )

        if save_to_file is not None:
            # 保存动画
            ani.save(
                save_to_file, writer="ffmpeg", fps=12, extra_args=["-vcodec", "libx264"]
            )
        else:
            # 展示动画
            plt.show()

    def show(self):
        """
        展示动画
        """
        self._render()

    def save(self, file_path: str):
        """
        保存动画

        :param file_path: 保存动画的文件路径
        """
        self._render(file_path)
