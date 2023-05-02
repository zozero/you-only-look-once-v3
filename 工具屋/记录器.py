import tensorflow as 张量流


class 记录器:
    def __init__(self, 日志目录名):
        # 日志文件夹的绝对路径不能出现中文路径
        self.书写者 = 张量流.summary.create_file_writer(日志目录名)

    def 标量的摘要(self, 标签, 值, 步子):
        with self.书写者.as_default():
            张量流.summary.scalar(标签, 值, step=步子)
            self.书写者.flush()

    def 列出标量的摘要(self, 标签与值的配对列表, 步子):
        with self.书写者.as_default():
            for 标签, 值 in 标签与值的配对列表:
                张量流.summary.scalar(标签, 值, step=步子)
            self.书写者.flush()
