import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 创建图形和轴
fig, ax = plt.subplots()

# 生成数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# 初始化线条
(line,) = ax.plot(x, y)


# 更新函数
def update(frame):
    # 更新 y 数据，相位随时间变化
    y = np.sin(x + frame * 0.1)
    line.set_ydata(y)
    return (line,)


time = 100


def gene():
    global time
    while time > 0:
        yield time
        time -= 1


# 创建动画
ani = animation.FuncAnimation(fig, update, interval=50, frames=gene, blit=True)

ani.save("animation.mp4", writer="ffmpeg", fps=12, extra_args=["-vcodec", "libx264"])
