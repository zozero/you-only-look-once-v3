import os

import torch

from 工具屋.工具库 import 载入分类列表
from 工具屋.解析配置库 import 解析数据配置
from 工具屋.记录器 import 记录器
from 配置屋.配置 import 参数

if __name__ == '__main__':
    记录者 = 记录器("日志房")

    设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("输出间", exist_ok=True)
    os.makedirs("检查点居室", exist_ok=True)

    数据配置 = 解析数据配置(参数.数据配置_文件路径)
    训练_路径 = 数据配置["训练"]
    验证_路径 = 数据配置["验证"]
    分类名称列表 = 载入分类列表(数据配置["名称列表"])
