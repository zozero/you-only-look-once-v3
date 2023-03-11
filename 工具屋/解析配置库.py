def 解析数据配置(路径):
    选项字典 = dict()
    选项字典['多个图形处理单元'] = '0,1,2,3'
    选项字典['线程数'] = '10'
    with open(路径, 'r', encoding='utf8') as 文件:
        多行 = 文件.readlines()
    for 行 in 多行:
        行 = 行.strip()
        if 行 == '' or 行.startswith('#'):
            continue
        键, 值 = 行.split('=')
        选项字典[键.strip()] = 值.strip()

    return 选项字典


def 解析模型配置(路径):
    文件 = open(路径, 'r')
    多行 = 文件.read().split('\n')
    多行 = [x for x in 多行 if x and not x.startswith('#')]
    多行 = [x.rstrip().lstrip() for x in 多行]
    模块定义列表 = []
    for 行 in 多行:
        if 行.startswith('['):
            模块定义列表.append({})
            模块定义列表[-1]['类型'] = 行[1:-1].rstrip()
            if 模块定义列表[-1]['类型'] == '卷积':
                模块定义列表[-1]['每批归一化'] = 0
        else:
            键, 值 = 行.split('=')
            值 = 值.strip()
            模块定义列表[-1][键.rstrip()] = 值.strip()

    return 模块定义列表
