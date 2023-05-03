# 采样数据集，连续值概念和二分值概念有不同的采样方法，要不可以选择只做其中一种（偏向于连续值概念）
# 目前完成了X的收集，但还需要标签y（即带入c函数得到的函数值）。

import pandas as pd
import random
import numpy as np
from typing import List, Tuple
from deduplicate import rotate1, rotate2, rotate3, symmetry, symrot1, symrot2, symrot3


def dedup(data):  # dedup要对trainset用而不是对原始的data数据集用
    memoryset = set()
    deduplicated_records = []

    for i in range(len(data)):
        record = data.iloc[i, :]
        # 从棋谱转换成棋盘，EMPTY=0，WHITE=1，BLACK=2
        record_map = []
        for row in record:
            for item in row:
                if item == -1:
                    item = 2
                record_map.append(item)

        record_map_str = ''.join(
            map(str, record_map))  # 转换成字符串用于对称旋转变换

        if record_map_str in memoryset:  # 判断是否重复
            continue
        deduplicated_records.append(record)
        memoryset.add(record_map_str)
        memoryset.add(rotate1(record_map_str))
        memoryset.add(rotate2(record_map_str))
        memoryset.add(rotate3(record_map_str))
        memoryset.add(symmetry(record_map_str))
        memoryset.add(symrot1(record_map_str))
        memoryset.add(symrot2(record_map_str))
        memoryset.add(symrot3(record_map_str))
    return deduplicated_records


# 将读出来的str类型转换为List[List[float]]类型，避开NaN部分
def data_preprocess(data: pd.DataFrame):
    na = np.where(data.isna())
    for i in range(len(data)):
        lst = [k for k in range(len(data.iloc[i, :]))]
        if i in set(na[0]):
            idx = [h for h, x in enumerate(na[0]) if x == i]
            for j in idx:
                lst.remove(na[1][j])
        for w in lst:
            data.iloc[i, w] = convert_cell(data.iloc[i, w])
    return data


def data_loader(data: pd.DataFrame, random_seed: int = 10):
    trainset = []
    valset = []
    testset = []  # 其实test可以随便取，不重复最好，一个棋谱中就可以取多条数据，不影响效果
    trainindexset = []
    valindexset = []
    testindexset = []

    na = np.where(data.isna())  # 二维

    for i in range(len(data)):

        # 这个过程有点冗余了，但暂时还没想到什么方法改进
        lst = [k for k in range(len(data.iloc[i, :]))]
        if i in set(na[0]):
            idx = [h for h, x in enumerate(na[0]) if x == i]
            for j in idx:
                lst.remove(na[1][j])  # 确保不会抽到空值

        random.seed(random_seed * len(data) + i)  # 确保随机但又能根据种子找回
        train_index = random.choice(lst)
        lst.remove(train_index)
        val_index = random.choice(lst)
        lst.remove(val_index)
        test_index = random.choice(lst)

        # data.iloc[i, train_index] = convert_cell(data.iloc[i, train_index])
        # data.iloc[i, val_index] = convert_cell(data.iloc[i, val_index])
        # data.iloc[i, test_index] = convert_cell(data.iloc[i, test_index])

        trainset.append(data.iloc[i, train_index])
        valset.append(data.iloc[i, val_index])
        testset.append(data.iloc[i, test_index])

        trainindexset.append(train_index)
        valindexset.append(val_index)
        testindexset.append(test_index)

    trainset = dedup(pd.DataFrame(trainset))
    valset = dedup(pd.DataFrame(valset))
    testset = dedup(pd.DataFrame(testset))

    return pd.DataFrame(trainset), pd.DataFrame(valset), pd.DataFrame(testset)


def convert_cell(cell: str):
    # 假设每个单元格内的数据格式如下: "[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]"
    # 移除方括号和空格
    cell = cell.strip("[] ").replace("], [", ";").split(";")
    # 将字符串转换为浮点数列表
    cell = [[int(num) for num in row.split(",")] for row in cell]
    return cell


if __name__ == '__main__':
    data = pd.read_csv('./Acquisition/allgamedata_small.csv', encoding='utf-8')
    data = data_preprocess(data)

    trainset, valset, testset = data_loader(data, 15)
    # 感觉valset是不需要的，可以在trainset中进行交叉验证
    # print(trainset.iloc[0, :])
    # print(trainset.iloc[2,0])
