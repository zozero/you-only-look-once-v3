def 解析数据配置(路径):
    选项字典 = dict()
    选项字典['多个图形处理单元'] = '0,1,2,3'
    选项字典['线程数'] = '10'
    with open(路径, 'r') as 文件:
        多行 = 文件.readlines()
    for 行 in 多行:
        行 = 行.strip()
        if 行 == '' or 行.startswith('#'):
            continue
        键, 值 = 行.split('=')
        选项字典[键.strip()] = 值.strip()

    return 选项字典
