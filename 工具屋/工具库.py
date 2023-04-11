import torch.nn.init


def 权重初始归一化(模型):
    类名 = 模型.__class__.__name__
    if 类名.find("卷积") != -1:
        torch.nn.init.normal_(模型.weight.data, 0.0, 0.02)
    elif 类名.find("二维批归一化") != -1:
        torch.nn.init.normal_(模型.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(模型.bias.data, 0.0)


def 载入分类列表(路径):
    文件 = open(路径, "r")
    名称列表 = 文件.read().split("\n")[:-1]
    return 名称列表


def 到中央处理器(张量):
    return 张量.detach().cpu()
