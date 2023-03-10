import tensorflow as 张量流


class 记录器:
    def __init__(self, 日志目录名):
        self.书写者 = 张量流.summary.create_file_writer(日志目录名)
