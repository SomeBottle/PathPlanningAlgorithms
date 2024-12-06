"""
可视化过程中能用到的一些工具函数
"""

# Matplotlib 图表中短边的长度（inch）
SHORTER_SIDE_LEN = 6


def scaled_figsize(m: float, n: float) -> tuple[float, float]:
    """
    按比例缩放，直至 m,n 中较小的一个到 SHORTER_SIDE_LEN

    :return: 缩放后的数值 (m,n)
    """
    smaller = min(m, n)
    factor = smaller / SHORTER_SIDE_LEN
    m /= factor
    n /= factor
    return (m, n)


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """
    把 16 进制颜色字符串转换为 RGB 值

    :param hex_str: 16 进制颜色字符串，比如 #FF00FF
    :return: RGB 值
    """
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
