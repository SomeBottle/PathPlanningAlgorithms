"""
异常类模块
"""


# 问题参数非法
class InvalidProblemArgument(Exception):

    def __init__(self, *args):
        super().__init__(*args)
