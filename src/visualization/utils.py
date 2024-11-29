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
