import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as 火炬函数
import torchvision.transforms as 变换
from PIL import Image

from 配置屋.增加物 import 水平翻转


def 填充到正方形(图片张量, 填充值):
    通道, 高, 宽 = 图片张量.shape
    尺寸的差 = np.abs(高 - 宽)
    填充1, 填充2 = 尺寸的差 // 2, 尺寸的差 - 尺寸的差 // 2
    # 在上左右下的位置填充
    填充 = (0, 0, 填充1, 填充2) if 高 <= 宽 else (填充1, 填充2, 0, 0)
    图片张量 = 火炬函数.pad(图片张量, 填充, "constant", value=填充值)

    return 图片张量, 填充


def 重置(图片张量, 尺寸):
    图片张量 = 火炬函数.interpolate(图片张量.unsqueeze(0), size=尺寸, mode="nearest").squeeze(0)
    return 图片张量


class 图片文件夹(Dataset):
    def __init__(self, 文件夹路径, 图片尺寸=416):
        super().__init__()
        self.文件列表 = sorted(glob.glob("%s/*.*" % 文件夹路径))
        self.图片尺寸 = 图片尺寸

    def __getitem__(self, 索引):
        图片路径 = self.文件列表[索引 % len(self.文件列表)]
        图片 = 变换.ToTensor()(Image.open(图片路径))
        图片, _ = 填充到正方形(图片, 0)
        图片 = 重置(图片, self.图片尺寸)
        return 图片路径, 图片

    def __len__(self):
        return len(self.文件列表)


class 数据集列表类(Dataset):
    def __init__(self, 列表文件路径, 图片尺寸=416, 是否增加=True, 是否多比例=True, 是否归一化标签=True):
        with open(列表文件路径, "r") as 文件:
            self.图片文件列表 = 文件.readlines()

        self.标签文件列表 = [
            路径.replace("图片箱子", "标签箱子").replace(".png", ".txt").replace(".jpg", ".txt") for 路径 in self.图片文件列表
        ]

        self.图片尺寸 = 图片尺寸
        self.最大物体数 = 100
        self.是否增加 = 是否增加
        self.是否多比例 = 是否多比例
        self.是否归一化标签 = 是否归一化标签
        self.最小尺寸 = self.图片尺寸 - 3 * 32
        self.最大尺寸 = self.图片尺寸 + 3 * 32
        self.批数 = 0

    def __getitem__(self, 索引):
        图片基础路径 = "./数据屋/coco"
        图片路径 = self.图片文件列表[索引 % len(self.图片文件列表)].rstrip()
        图片路径 = 图片基础路径 + 图片路径
        # 将图提取为张量
        图片张量 = 变换.ToTensor()(Image.open(图片路径).convert('RGB'))

        # 处理少于三个通道的图片
        if len(图片张量.shape) != 3:
            图片张量 = 图片张量.unsqueeze(0)
            图片张量 = 图片张量.expand((3, 图片张量.shape[1:]))

        # 填充图片成为正方形的图片
        _, 高, 宽 = 图片张量.shape
        高_因子, 宽_因子 = (高, 宽) if self.是否归一化标签 else (1, 1)
        图片张量, 填充 = 填充到正方形(图片张量, 0)
        _, 填充后的高, 填充后的宽 = 图片张量.shape

        # ---------
        #  标签
        # ---------
        标签文件路径 = self.标签文件列表[索引 % len(self.图片文件列表)].rstrip()
        标签文件路径 = 图片基础路径 + 标签文件路径
        目标列表 = None
        # 了解以下代码的用处，https://cocodataset.org/#people
        if os.path.exists(标签文件路径):
            盒形张量列表 = torch.from_numpy(np.loadtxt(标签文件路径).reshape(-1, 5))
            x1 = 宽_因子 * (盒形张量列表[:, 1] - 盒形张量列表[:, 3] / 2)
            y1 = 高_因子 * (盒形张量列表[:, 2] - 盒形张量列表[:, 4] / 2)
            x2 = 宽_因子 * (盒形张量列表[:, 1] + 盒形张量列表[:, 3] / 2)
            y2 = 高_因子 * (盒形张量列表[:, 2] + 盒形张量列表[:, 4] / 2)

            x1 += 填充[0]
            y1 += 填充[2]
            x2 += 填充[1]
            y2 += 填充[3]

            盒形张量列表[:, 1] = ((x1 + x2) / 2) / 填充后的宽
            盒形张量列表[:, 2] = ((y1 + y2) / 2) / 填充后的高
            盒形张量列表[:, 3] *= 宽_因子 / 填充后的宽
            盒形张量列表[:, 4] *= 高_因子 / 填充后的高

            目标列表 = torch.zeros((len(盒形张量列表), 6))
            # 占位符 类别  中心点x坐标 中心点y坐标
            目标列表[:, 1:] = 盒形张量列表

        if self.是否增加:
            if np.random.random() < 0.5:
                图片张量, 目标列表 = 水平翻转(图片张量, 目标列表)

        return 图片路径, 图片张量, 目标列表

    def 整理用函数(self, 一批数据):
        路径列表, 图片列表, 目标列表 = list(zip(*一批数据))
        目标列表 = [盒形列表 for 盒形列表 in 目标列表 if 盒形列表 is not None]
        # 添加样本索引到目标列表
        for 索引, 盒形列表 in enumerate(目标列表):
            盒形列表[:, 0] = 索引
        目标列表 = torch.cat(目标列表, 0)

        # 每十批选择新的图片尺寸
        if self.是否多比例 and self.批数 % 10 == 0:
            self.图片尺寸 = random.choice(range(self.最小尺寸, self.最大尺寸 + 1, 32))

        图片列表 = torch.stack([重置(图片张量, self.图片尺寸) for 图片张量 in 图片列表])
        self.批数 += 1
        return 路径列表, 图片列表, 目标列表

    def __len__(self):
        return len(self.图片文件列表)
