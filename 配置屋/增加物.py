import torch


def 水平翻转(图片张量, 目标列表):
    图片张量 = torch.flip(图片张量, [-1])
    目标列表[:, 2] = 1 - 目标列表[:, 2]
    return 图片张量, 目标列表
