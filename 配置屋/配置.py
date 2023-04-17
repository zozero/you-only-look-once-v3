import argparse

解析器 = argparse.ArgumentParser()
解析器.add_argument("--轮回数", type=int, default=100, help="轮回的次数")
解析器.add_argument("--单批数", type=int, default=4, help="一批图片的张数")
解析器.add_argument("--梯度累积", type=int, default=2, help="每个步骤前的梯度累积数")
解析器.add_argument("--模型定义_文件路径", type=str, default="配置屋/我只看一次版本3.配置", help="模型配置文件路径")
解析器.add_argument("--数据配置_文件路径", type=str, default="配置屋/coco.数据", help="数据配置文件路径")
解析器.add_argument("--预训练权重_文件路径", type=str, help="如果使用检查点内的模型的话")
解析器.add_argument("--中央处理器的线程数", type=int, default=0, help="计算单批数据期间使用中央处理器的线程数量")
解析器.add_argument("--图片尺寸", type=int, default=416, help="每张图片的尺寸")
解析器.add_argument("--检查点间隔", type=int, default=1, help="保存模型权重的间隔")
解析器.add_argument("--评估的间隔", type=int, default=1, help="验证集评估时的间隔")
解析器.add_argument("--是否计算均值平均准确率", type=bool, default=False,
                    help="如果 True 每十个批次计算一次均值平均准确率")
解析器.add_argument("--允许多尺寸训练", type=bool, default=True, help="允许多尺寸训练")
参数 = 解析器.parse_args()
print(参数)

