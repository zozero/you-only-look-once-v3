import os
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from 工具屋.工具库 import 载入分类列表, 权重初始归一化
from 工具屋.解析配置库 import 解析数据配置
from 工具屋.记录器 import 记录器
from 模型库 import 黑夜网络
from 配置屋.数据处理 import 数据集列表类
from 配置屋.配置 import 参数

if __name__ == '__main__':
    # 记录者 = 记录器("日志房")

    设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("输出间", exist_ok=True)
    os.makedirs("检查点居室", exist_ok=True)

    数据配置 = 解析数据配置(参数.数据配置_文件路径)
    训练_路径 = 数据配置["训练"]
    验证_路径 = 数据配置["验证"]
    分类名称列表 = 载入分类列表(数据配置["名称列表"])

    模型 = 黑夜网络(参数.模型定义_文件路径).to(设备)
    模型.apply(权重初始归一化)

    if 参数.预训练权重_文件路径:
        if 参数.预训练权重_文件路径.endswith(".pth"):
            模型.load_state_dict(torch.load(参数.预训练权重_文件路径))
        else:
            模型.载入黑夜网络权重(参数.预训练权重_文件路径)

    训练用数据集 = 数据集列表类(训练_路径, 是否增加=True, 是否多比例=参数.允许多尺寸训练)
    训练用数据加载器 = torch.utils.data.DataLoader(
        训练用数据集,
        batch_size=参数.单批数,
        shuffle=True,
        num_workers=参数.中央处理器的线程数,
        pin_memory=True,
        collate_fn=训练用数据集.整理用函数
    )

    优化器 = torch.optim.Adam(模型.parameters())

    指标 = [
        "网格尺寸",
        "损失值",
        "x",
        "y",
        "宽",
        "高",
        "置信度",
        # 暂时不能明确其意
        "类别",
        "类别_准确度",
        "召回率50",
        "召回率75",
        "精确度",
        "有对象的置信度",
        "无对象的置信度",
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
