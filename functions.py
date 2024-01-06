import os

# 该函数用以快速生成读取数据集需要的参数
def Easyget_path_and_windowsize_list(root_dir = os.getcwd(), suffix=None):
    """
    只针对该数据集，快速获取读取数据集需要的参数
    :param root_dir: 搜索的根目录
    :param suffix: 搜索的关键字
    :return: tuple,分别是数据集绝对路径的列表，以及建议窗口尺寸的列表
    """
    if suffix is None:
        suffix = ['m1', 'm2', 'm3']
    path_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname[-2:] in suffix:
                path_list.append(os.path.abspath(os.path.join(dirpath, dirname)))

    windowsize_list = []
    for path in path_list:
        if path.split('\\')[-3] == '6400_embedded':
            windowsize_list.append(1024)
        elif path.split('\\')[-3] == '10000_recorder':
            windowsize_list.append(1000)
    return path_list,windowsize_list




