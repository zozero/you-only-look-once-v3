import datetime
import os
import random
import time

import numpy as np
from terminaltables import AsciiTable
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from 工具屋.工具库 import 载入分类列表, 权重初始归一化
from 工具屋.解析配置库 import 解析数据配置
from 工具屋.记录器 import 记录器
from 模型库 import 黑夜网络
from 测试 import 评估
from 配置屋.数据处理 import 数据集列表类
from 配置屋.配置 import 参数

# np.random.seed(1)
# random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
#
# 可用来启用数据调试，看到具体数据，这是在使用多个gpu时使用
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    # 记录者 = 记录器("D:/BaiduNetdiskDownload/log")

    设备 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs("输出间", exist_ok=True)
    os.makedirs("检查点居室", exist_ok=True)

    数据配置 = 解析数据配置(参数.数据配置_文件路径)
    训练_路径 = 数据配置["训练"]
    验证_路径 = 数据配置["验证"]
    分类名称列表 = 载入分类列表(数据配置["名称列表"])

    训练用数据集 = 数据集列表类(训练_路径, 是否增加=True, 是否多比例=参数.允许多尺寸训练)
    训练用数据加载器 = torch.utils.data.DataLoader(
        训练用数据集,
        batch_size=参数.单批数,
        shuffle=True,
        num_workers=参数.中央处理器的线程数,
        pin_memory=True,
        collate_fn=训练用数据集.整理用函数
    )

    模型 = 黑夜网络(参数.模型定义_文件路径)
    # if torch.cuda.device_count() > 1:
    #     print("让我们使用", torch.cuda.device_count(), "图像处理单元吧!")
    #     模型 = nn.DataParallel(模型)
    模型.to(设备)
    模型.apply(权重初始归一化)
    if 参数.预训练权重_文件路径:
        if 参数.预训练权重_文件路径.endswith(".pth"):
            模型.load_state_dict(torch.load(参数.预训练权重_文件路径))
        else:
            模型.载入黑夜网络权重(参数.预训练权重_文件路径)

    优化器 = torch.optim.Adam(模型.parameters())

    # 各种类型的损失值指标以及其他指标
    指标列表 = [
        "网格尺寸",
        "损失值",
        "x",
        "y",
        "宽",
        "高",
        "置信度",
        "分类",
        "分类的准确度",
        "召回率50",
        "召回率75",
        "精确度",
        "有物体的置信度",
        "无物体的置信度"
    ]

    for 轮回 in range(参数.轮回数):
        模型.train()
        起始时间 = time.time()
        for 批索引, (_, 图片列表, 目标列表) in enumerate(训练用数据加载器):
            已训批数 = len(训练用数据加载器) * 轮回 + 批索引

            图片列表 = Variable(图片列表.to(设备))
            目标列表 = Variable(目标列表.to(设备), requires_grad=False)
            print("图片列表", 图片列表.shape)
            print("目标列表", 目标列表.shape)
            损失值, 输出列表 = 模型(图片列表, 目标列表)
            损失值.backward()

            if 已训批数 % 参数.梯度累积数:
                优化器.step()
                优化器.zero_grad()

            # ----------------
            #   日志进展
            # ----------------

            日志字符串 = "\n---- [轮回 %d/%d，批数 %d/%d] ----\n" % (轮回, 参数.轮回数, 批索引, len(训练用数据加载器))
            指标表格 = [["指标列表", *[f"我只看一次层{索引}" for 索引 in range(len(模型.我只看一次层列表))]]]

            for 索引, 指标 in enumerate(指标列表):
                格式字典 = {键名: "%.6f" for 键名 in 指标列表}
                格式字典["网格尺寸"] = "%2d"
                格式字典["分类的准确度"] = "%.2f%%"
                原生指标列表 = [格式字典[指标] % 我只看一次.指标字典.get(指标, 0) for 我只看一次 in 模型.我只看一次层列表]
                指标表格 += [[指标, *原生指标列表]]

                张量仪表盘日志 = []
                for 序号, 我只看一次 in enumerate(模型.我只看一次层列表):
                    for 名称, 指标值 in 我只看一次.指标字典.items():
                        if 名称 != "网格尺寸":
                            张量仪表盘日志 += [(f"{名称}_{序号 + 1}", 指标值)]
                张量仪表盘日志 += [("损失值", 损失值.item())]
                # 记录者.列出标量的摘要(张量仪表盘日志, 已训批数)

            日志字符串 += AsciiTable(指标表格).table
            日志字符串 += f"\n整体损失值 {损失值.item()}"

            轮回剩余批数 = len(训练用数据加载器) - (批索引 + 1)
            剩余时间 = datetime.timedelta(seconds=轮回剩余批数 * (time.time() - 起始时间) / (批索引 + 1))
            日志字符串 += f"\n---- 剩余时间 {剩余时间}"

            print(日志字符串)

        if 轮回 % 参数.检查点间隔 == 0:
            print("---- 正在保存模型 ----")
            torch.save(模型.state_dict(), f"检查点居室/我只看一次版本3_检查点_%d.pth" % 轮回)

        if 轮回 % 参数.评估的间隔 == 0:
            print("---- 正在评估模型 ----")
            # 在验证集上评估模型
            # 指标f1中的f是指某个人名 Ronald Fisher，这个人提出了f分布，https://stats.stackexchange.com/questions/300975/why-is-f-score-called-f-score
            精确度, 召回率, 平均精确度, 指标f1, 平均精确度_分类 = 评估(模型, 路径=验证_路径, 交并比阈值=0.5, 置信度阈值=0.5, 非极大值抑制阈值=0.5, 图片尺寸=参数.图片尺寸, 单批数=8)
            评估的指标 = [
                ('验证_精确度', 精确度.mean()),
                ('验证_召回率', 召回率.mean()),
                ('验证_均值平均精确度', 平均精确度.mean()),
                ('验证_指标f1', 指标f1.mean())
            ]
            # 记录者.列出标量的摘要(评估的指标, 轮回)

            平均精确度_表格 = [["索引", "分类名", "平均精确度"]]
            for 索引, 分类 in enumerate(平均精确度_分类):
                平均精确度_表格 += [[分类, 分类名称列表[分类], "%.5f" % 平均精确度[索引]]]
            print(AsciiTable(平均精确度_表格).table)
            print(f"--- 均值平均精确度{平均精确度.mean()}")
