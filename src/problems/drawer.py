"""
问题绘制模块
"""

import tkinter as tk
import numpy as np

from .cell_status import CellStatus
from .problem import Problem
from .utils import close_problem_obstacles

START_POINT_COLOR = "#D91656"
END_POINT_COLOR = "#80C4E9"
LINE_WIDTH = 2  # 绘制线条的宽度

# 图像短边至少要有多长
MIN_SIDE_LENGTH = 50


def draw_problem(width: int, height: int) -> Problem:
    """
    直接手动绘制问题

    :param width: 地图宽度
    :param height: 地图高度
    :return: 生成的寻路问题 Problem 对象
    """

    # width, height 太小
    if min(width, height) < MIN_SIDE_LENGTH:
        raise ValueError(f"width and height should be at lease {MIN_SIDE_LENGTH}")

    # 主窗口
    root = tk.Tk()
    root.title("Pixel Drawing Board")

    # 画布大小
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack()

    # 用于保存绘制的像素，即阻塞物
    pixels = np.zeros((height, width), dtype=np.uint8)

    # 开始点和结束点 (i,j)
    start_pos = (0, 0)
    end_pos = (height - 1, width - 1)

    # 用于保存鼠标按下时的坐标
    last_x = 0
    last_y = 0

    # 提示文本
    label = tk.Label(root, text="Draw obstacles please.")
    label.pack()

    # 绘制时鼠标按下事件
    def draw_mouse_down(event):
        nonlocal last_x, last_y
        last_x, last_y = event.x, event.y

    # 绘制时鼠标移动事件
    def draw_mouse_move(event):
        nonlocal last_x, last_y
        x, y = event.x, event.y
        canvas.create_line((last_x, last_y, x, y), fill="black", width=LINE_WIDTH)
        # 这里切片注意切片必须是 start:end 中 start<end
        y_range_start = min(last_y, y)
        y_range_end = max(last_y, y)
        x_range_start = min(last_x, x)
        x_range_end = max(last_x, x)
        y_extra = 0
        x_extra = 0
        if y_range_end - y_range_start > x_range_end - x_range_start:
            # y 方向变得更多，那么加粗线条就在 x 方向
            x_extra = LINE_WIDTH
        else:
            y_extra = LINE_WIDTH
        pixels[
            y_range_start : y_range_end + y_extra,
            x_range_start : x_range_end + x_extra,
        ] = 1  # 更新二维数组，1 代表此处有阻塞物
        # print(x, y)
        last_x, last_y = x, y

    # 绑定鼠标事件
    canvas.bind("<Button-1>", draw_mouse_down)
    canvas.bind("<B1-Motion>", draw_mouse_move)

    # 保存图像和二维数组的按钮
    def start_selecting_point():
        # 取消展示按钮
        ok_button.pack_forget()
        label.configure(text="Select start point please.")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<Button-1>")
        canvas.bind("<Motion>", point_move_move)
        canvas.bind("<Button-1>", select_start_point)
        canvas.configure(cursor="none")

    # 记录上一个绘制的像素
    previous_pixel = None
    point_pixel_color = START_POINT_COLOR

    # 选好了起始坐标
    def select_start_point(event):
        nonlocal start_pos, point_pixel_color
        x, y = event.x, event.y
        start_pos = (y, x)  # 注意 start_pos 存的是列号和行号
        canvas.unbind("<Button-1>")
        canvas.create_rectangle(
            (x, y, x + 4, y + 4), fill=point_pixel_color, outline=""
        )
        # 接下来选择终点
        point_pixel_color = END_POINT_COLOR
        label.configure(text="Select end point please.")
        canvas.bind("<Button-1>", select_end_point)

    def select_end_point(event):
        nonlocal end_pos
        x, y = event.x, event.y
        end_pos = (y, x)
        canvas.unbind("<Button-1>")
        # 结束绘制过程
        root.destroy()

    # 绘制点时的鼠标移动事件
    def point_move_move(event):
        nonlocal previous_pixel
        if previous_pixel is not None:
            # 擦除掉上一个像素，模拟像素跟随鼠标移动的效果
            canvas.delete(previous_pixel)
        x, y = event.x, event.y
        previous_pixel = canvas.create_rectangle(
            (x, y, x + 4, y + 4), fill=point_pixel_color, outline=""
        )

    ok_button = tk.Button(root, text="OK", command=start_selecting_point)
    ok_button.pack()

    # 鼠标初始为画笔状
    canvas.configure(cursor="pencil")

    # 运行主循环
    root.mainloop()

    # 把 0/1 矩阵替换为枚举值矩阵
    pixels = np.where(pixels == 0, CellStatus.EMPTY, CellStatus.BLOCKED)
    # 填充对角处的障碍物
    # 转换为 Problem 对象
    return Problem(close_problem_obstacles(pixels.tolist()), start_pos, end_pos)
