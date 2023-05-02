import argparse
import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from matplotlib import patches
from matplotlib.ticker import NullLocator
from torch import nn
from torch.autograd import Variable
import tqdm as 进度条
from torch.utils.data import DataLoader

from 工具屋.工具库 import 坐标和宽高转坐标和坐标, 统计并获取某批的指标数据, 计算每批分类平均精确度, 非极大值抑制, 载入分类列表, 恢复盒子列表正常比例
from 模型库 import 黑夜网络
from 配置屋.数据处理 import 数据集列表类, 图片文件夹


def 评估(模型: 黑夜网络, 路径, 交并比阈值, 置信度阈值, 非极大值抑制阈值, 图片尺寸, 单批数):
    模型.eval()

    数据集 = 数据集列表类(路径, 图片尺寸=图片尺寸, 是否增加=False, 是否多比例=False)
    数据加载器 = torch.utils.data.DataLoader(数据集, batch_size=单批数, shuffle=False, num_workers=1, collate_fn=数据集.整理用函数)

    浮点型张量 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    标签列表 = []
    样本指标列表 = []
    for 批索引, (_, 图片列表, 目标列表) in enumerate(进度条.tqdm(数据加载器, desc="检测物体列表")):
        标签列表 += 目标列表[:, 1].tolist()
        目标列表[:, 2:] = 坐标和宽高转坐标和坐标(目标列表[:, 2:])
        目标列表[:, 2:] *= 图片尺寸

        图片列表 = Variable(图片列表.type(浮点型张量), requires_grad=False)

        with torch.no_grad():
            输出列表 = 模型(图片列表)
            输出列表 = 非极大值抑制(输出列表, 置信度阈值=置信度阈值, 非极大值抑制阈值=非极大值抑制阈值)

        样本指标列表 += 统计并获取某批的指标数据(输出列表, 目标列表, 交并比阈值)

    有物体判断为有物体, 预测的分数列表, 预测的标签列表 = [np.concatenate(某, 0) for 某 in list(zip(*样本指标列表))]
    精确度, 召回率, 平均精确度, 指标f1, 平均精确度_分类 = 计算每批分类平均精确度(有物体判断为有物体, 预测的分数列表, 预测的标签列表, 标签列表)

    return 精确度, 召回率, 平均精确度, 指标f1, 平均精确度_分类


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--图片文件夹", type=str, default="数据屋/样例", help="数据集路径")
    parser.add_argument("--定义模型的文件", type=str, default="配置屋/我只看一次版本3.配置", help="定义模型的配置文件")
    parser.add_argument("--权重文件路径", type=str, default="检查点居室/我只看一次版本3_检查点_0.pth", help="权重文件路径")
    parser.add_argument("--分类名称文件路径", type=str, default="配置屋/coco.名称列表", help="分类名称文件路径")
    parser.add_argument("--置信度阈值", type=float, default=0.8, help="物体置信度阈值")
    parser.add_argument("--非极大值抑制阈值", type=float, default=0.4, help="非极大值抑制阈值")
    parser.add_argument("--单批数", type=int, default=1, help="每批的图片数量")
    parser.add_argument("--中央处理器数量", type=int, default=0, help="生成每批数据期间使用多少个中央处理器")
    parser.add_argument("--图片尺寸", type=int, default=416, help="每张图片维度的尺寸")
    parser.add_argument("--模型的检查点", type=str, help="模型检查点的路径")
    参数 = parser.parse_args()
    print(参数)

    设备 = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    os.makedirs("输出间", exist_ok=True)

    # 设置模型
    模型 = 黑夜网络(参数.定义模型的文件, 图片尺寸=参数.图片尺寸)
    # if torch.cuda.device_count() > 1:
    #     print("让我们使用", torch.cuda.device_count(), "图像处理单元吧!")
    #     模型 = nn.DataParallel(模型)
    模型.to(设备)

    if 参数.权重文件路径.endswith(".weights"):
        模型.载入黑夜网络权重(参数.权重文件路径)
    else:
        模型.load_state_dict(torch.load(参数.权重文件路径))

    # 设置为评估模型
    模型.eval()

    数据加载器 = DataLoader(
        图片文件夹(参数.图片文件夹, 图片尺寸=参数.图片尺寸),
        batch_size=参数.单批数,
        shuffle=False,
        num_workers=参数.中央处理器数量
    )

    分类列表 = 载入分类列表(参数.分类名称文件路径)

    浮点型张量 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    图片列表 = []
    图片检测列表 = []

    print("\n执行物体检测任务：")
    开始时间 = time.time()
    for 批索引, (图片路径列表, 输入的图片列表) in enumerate(数据加载器):
        输入的图片列表 = Variable(输入的图片列表.type(浮点型张量).to(设备))

        with torch.no_grad():
            检测结果列表 = 模型(输入的图片列表)
            检测结果列表 = 非极大值抑制(检测结果列表, 参数.置信度阈值, 参数.非极大值抑制阈值)

        当前时间 = time.time()
        计算用时 = datetime.timedelta(seconds=当前时间 - 开始时间)
        开始时间 = 当前时间
        print("\t+ 批索引：%d，计算用时：%s" % (批索引, 计算用时))

        图片列表.extend(图片路径列表)
        图片检测列表.extend(检测结果列表)

    颜色位图 = plt.get_cmap("tab20b")
    颜色列表 = [颜色位图(索引) for 索引 in np.linspace(0, 1, 20)]

    print("\n正在保存的图片列表：")
    for 图片索引, (路径, 检测结果列表) in enumerate(zip(图片列表, 图片检测列表)):
        print("第%d张图片：'%s'" % (图片索引, 路径))
        图片 = np.array(Image.open(路径))
        plt.figure()
        图像, x轴 = plt.subplots(1)
        x轴.imshow(图片)

        if 检测结果列表 is not None:
            检测结果列表 = 恢复盒子列表正常比例(检测结果列表, 参数.图片尺寸, 图片.shape[:2])
            唯一标签列表 = 检测结果列表[:, -1].cpu().unique()
            预测的分类数量 = len(唯一标签列表)

            盒子边框的颜色 = random.sample(颜色列表, 预测的分类数量)

            for x1, y1, x2, y2, 置信度, 分类置信度, 预测的分类 in 检测结果列表:
                print("\t+ 标签：%s，置信度：%.5f" % (分类列表[int(预测的分类)], 分类置信度.item()))

                盒子的宽 = x2 - x1
                盒子的高 = y2 - y1
                颜色 = 盒子边框的颜色[int(np.where(唯一标签列表 == int(预测的分类))[0])]
                盒子边框 = patches.Rectangle((x1, y1), 盒子的宽, 盒子的高, linewidth=2, edgecolor=颜色, facecolor="none")
                x轴.add_patch(盒子边框)
                plt.text(x1, y1, s=分类列表[int(预测的分类)], color="white", verticalalignment="top", bbox={"color": 颜色, "pad": 0})

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        图像文件名 = 路径.split("/")[-1].split(".")[0]
        plt.savefig(f"输出间/{图像文件名}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
