def 载入分类列表(路径):
    文件 = open(路径, "r")
    名称列表 = 文件.read().split("\n")[:-1]
    return 名称列表
