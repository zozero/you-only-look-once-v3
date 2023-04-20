import tensorflow as 张量流


class 记录器:
    def __init__(self, 日志目录名):
        # 日志文件夹的绝对路径不能出现中文路径
        self.书写者 = 张量流.summary.create_file_writer(日志目录名)
