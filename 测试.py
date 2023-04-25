import numpy as np
import torch.utils.data
from torch.autograd import Variable
import tqdm as 进度条

from 工具屋.工具库 import 坐标和宽高转坐标和坐标, 统计并获取某批的指标数据, 计算每批分类平均精确度, 非极大值抑制
from 模型库 import 黑夜网络
from 配置屋.数据处理 import 数据集列表类


def 评估(模型: 黑夜网络, 路径, 交并比阈值, 置信度阈值, 非极大值抑制阈值, 图片尺寸, 单批数):
    模型.eval()

    数据集 = 数据集列表类(路径, 图片尺寸=图片尺寸, 是否增加=False, 是否多比例=False)
    数据加载器 = torch.utils.data.DataLoader(数据集, batch_size=单批数, shuffle=False, num_workers=1, collate_fn=数据集.整理用函数)

    浮点型张量 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    标签列表 = []
    样本指标列表 = []
    for 批索引, (_, 图片列表, 目标列表) in enumerate(进度条.tqdm(数据加载器, desc="检测对象列表")):
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
