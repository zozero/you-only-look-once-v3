import os

import torch

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

    数据集 = 数据集列表类(训练_路径,是否增加=True,是否多比例=参数.允许多尺寸训练)
    数据集.__getitem__(1)
