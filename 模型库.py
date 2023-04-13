import numpy as np
import torch
from torch import nn
import torch.nn.functional as 火炬函数

from 工具屋.工具库 import 到中央处理器
from 工具屋.解析配置库 import 解析模型配置


def 创建模块(模块定义列表):
    超参数列表 = 模块定义列表.pop(0)
    输出_过滤器_列表 = [int(超参数列表["通道数"])]
    模块列表 = nn.ModuleList()
    for 模块_索引, 模块_定义 in enumerate(模块定义列表):
        多个模块 = nn.Sequential()

        if 模块_定义["类型"] == "卷积":
            是否批归一化 = int(模块_定义["是否归一化批"])
            过滤器数量 = int(模块_定义["过滤器数量"])
            内核大小 = int(模块_定义["大小"])
            填充 = (内核大小 - 1) // 2
            多个模块.add_module(
                f"卷积_{模块_索引}",
                nn.Conv2d(
                    in_channels=输出_过滤器_列表[-1],
                    out_channels=过滤器数量,
                    kernel_size=内核大小,
                    stride=int(模块_定义["步长"]),
                    padding=填充,
                    bias=not 是否批归一化
                )
            )

            if 是否批归一化:
                多个模块.add_module(f"批归一化_{模块_索引}", nn.BatchNorm2d(过滤器数量, momentum=0.9, eps=1e-5))
            if 模块_定义["激活函数"] == "泄露型线性整流函数":
                多个模块.add_module(f"泄露型_{模块_索引}", nn.LeakyReLU(0.1))
        elif 模块_定义["类型"] == "最大池化":
            内核大小 = int(模块_定义["大小"])
            步长 = int(模块_定义["步长"])
            if 内核大小 == 2 and 步长 == 1:
                多个模块.add_module(f"_调试_填充_{模块_索引}", nn.ZeroPad2d((0, 1, 0, 1)))
            最大池化 = nn.MaxPool2d(kernel_size=内核大小, stride=步长, padding=int((内核大小 - 1) // 2))
            多个模块.add_module(f"最大池化_{模块_索引}", 最大池化)
        elif 模块_定义["类型"] == "上采样":
            某个上采样 = 上采样(比例因子=int(模块_定义["步长"]), 模式="nearest")
            多个模块.add_module(f"上采样{模块_索引}", 某个上采样)
        elif 模块_定义["类型"] == "路径":
            层数列表 = [int(x) for x in 模块_定义["层数"].split(",")]
            过滤器数量 = sum([输出_过滤器_列表[1:][i] for i in 层数列表])
            多个模块.add_module(f"路径_{模块_索引}", 空层())
        elif 模块_定义["类型"] == "捷径":
            过滤器数量 = 输出_过滤器_列表[1:][int(模块_定义["来自"])]
            多个模块.add_module(f"捷径_{模块_索引}", 空层())
        elif 模块_定义["类型"] == "我只看一次":
            锚定盒_序号列表 = [int(x) for x in 模块_定义["掩码"].split(",")]
            锚定盒 = [int(x) for x in 模块_定义["锚定盒"].split(",")]
            锚定盒 = [(锚定盒[i], 锚定盒[i + 1]) for i in range(0, len(锚定盒), 2)]
            锚定盒 = [锚定盒[i] for i in 锚定盒_序号列表]
            分类数 = int(模块_定义["分类数"])
            图片尺寸 = int(超参数列表["高度"])

            一个我只看一次层 = 我只看一次层(锚定盒, 分类数, 图片维度=图片尺寸)
            多个模块.add_module(f"我只看一次_{模块_索引}", 一个我只看一次层)

        模块列表.append(多个模块)
        # noinspection PyUnboundLocalVariable
        输出_过滤器_列表.append(过滤器数量)

    return 超参数列表, 模块列表


class 上采样(nn.Module):
    def __init__(self, 比例因子, 模式="nearest"):
        super().__init__()
        self.比例因子 = 比例因子
        self.模式 = 模式

    def forward(self, 输入):
        输出 = 火炬函数.interpolate(输入, scale_factor=self.比例因子, mode=self.模式)
        return 输出


class 空层(nn.Module):
    def __init__(self):
        super().__init__()


class 我只看一次层(nn.Module):
    def __init__(self, 锚定盒, 分类数, 图片维度=416):
        super().__init__()
        self.锚定盒 = 锚定盒
        self.锚定盒数量 = len(锚定盒)
        self.分类数 = 分类数
        self.忽视_阈值 = 0.5
        self.均方误差损失值函数 = nn.MSELoss()
        self.二元交叉熵损失值函数 = nn.BCELoss()
        self.对象_比例 = 1
        self.无对象_比例 = 100
        self.指标 = {}
        self.图片维度 = 图片维度
        self.网格_尺寸 = 0

    def forward(self, 输入列表, 目标列表=None, 图片维度=None):
        print(输入列表.shape)
        浮点型张量 = torch.cuda.FloatTensor if 输入列表.is_cuda else torch.FloatTensor
        长整型张量 = torch.cuda.LongTensor if 输入列表.is_cuda else torch.LongTensor
        字节型张量 = torch.cuda.ByteTensor if 输入列表.is_cuda else torch.ByteTensor
        exit()


class 黑夜网络(nn.Module):
    def __init__(self, 配置文件路径, 图片尺寸=416):
        super().__init__()
        self.模块定义列表 = 解析模型配置(配置文件路径)
        self.超参数列表, self.模块列表 = 创建模块(self.模块定义列表)
        self.图片尺寸 = 图片尺寸
        self.见过 = 0
        self.头部_信息 = np.array([0, 0, 0, self.见过, 0], dtype=np.int32)

    def forward(self, 输入列表, 目标列表=None):
        图片维度 = 输入列表.shape[2]
        损失值 = 0
        层的输出列表, 我只看一次层的输出列表 = [], []
        for 索引, (模块_定义, 模块) in enumerate(zip(self.模块定义列表, self.模块列表)):
            if 模块_定义["类型"] in ["卷积", "上采样", "最大池化"]:
                输入列表 = 模块(输入列表)
            elif 模块_定义["类型"] == "路径":
                输入列表 = torch.cat([层的输出列表[int(层索引)] for 层索引 in 模块_定义["层数"].split(",")], 1)
            elif 模块_定义["类型"] == "捷径":
                层索引 = int(模块_定义["来自"])
                输入列表 = 层的输出列表[-1] + 层的输出列表[层索引]
            elif 模块_定义["类型"] == "我只看一次":
                输入列表, 层损失值 = 模块[0](输入列表, 目标列表, 图片维度)
                损失值 += 层损失值
                我只看一次层的输出列表.append(输入列表)
            层的输出列表.append(输入列表)
        我只看一次层的输出列表 = 到中央处理器(torch.cat(我只看一次层的输出列表, 1))

        return 我只看一次层的输出列表 if 目标列表 is None else (损失值, 层的输出列表)

    def 载入黑夜网络权重(self, 权重路径):
        with open(权重路径, "rb") as 文件:
            头部 = np.fromfile(文件, dtype=np.int32, count=5)
            self.头部_信息 = 头部
            self.见过 = 头部[3]
            权重数组 = np.fromfile(文件, dtype=np.flot32)

        # 载入骨干权重来建立近路
        近路 = None
        if "黑夜网络53.卷积.74" in 权重路径:
            近路 = 75

        指针 = 0
        for 索引, (模块定义, 模块) in enumerate(zip(self.模块定义列表, self.模块列表)):
            if 索引 == 近路:
                break

            if 模块定义["类型"] == "卷积":
                卷积层 = 模块[0]
                if 模块定义["批归一化"]:
                    批归一化层 = 模块[1]
                    偏置项数目 = 批归一化层.bias.numel()
                    # 偏置项
                    批归一化的偏置项 = torch.from_numpy(权重数组[指针:指针 + 偏置项数目]).view_as(批归一化层.bias)
                    批归一化层.bias.data.copy_(批归一化的偏置项)
                    指针 += 偏置项数目
                    # 权重
                    批归一化的权重 = torch.from_numpy(权重数组[指针:指针 + 偏置项数目]).view_as(批归一化层.weight)
                    批归一化层.bias.weight.copy_(批归一化的权重)
                    指针 += 偏置项数目
                    # 运行时平均数
                    批归一化运行时平均数 = torch.from_numpy(权重数组[指针:指针 + 偏置项数目]).view_as(
                        批归一化层.running_mean)
                    批归一化层.bias.running_mean.copy_(批归一化运行时平均数)
                    指针 += 偏置项数目
                    # 运行时方差
                    批归一化运行时方差 = torch.from_numpy(权重数组[指针:指针 + 偏置项数目]).view_as(
                        批归一化层.running_var)
                    批归一化层.running_var.data.copy_(批归一化运行时方差)
                    指针 += 偏置项数目
                else:
                    偏置项数目 = 卷积层.bias.numel()
                    卷积层的偏置项 = torch.from_numpy(权重数组[指针:指针 + 偏置项数目]).view_as(卷积层.bias)
                    卷积层.bias.data.copy_(卷积层的偏置项)
                    指针 += 偏置项数目

                权重的数目 = 卷积层.weight.numel()
                卷积层的权重 = torch.from_numpy(权重数组[指针:指针 + 权重的数目]).view_as(卷积层.weight)
                卷积层.weight.data.copy_(卷积层的权重)
                指针 += 偏置项数目
