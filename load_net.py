import torch
import sys
from resnet import ResNet
from load_data import data_preprocess, data_loader, load_dataframe
import pandas as pd
from player_color import BLACK, WHITE, EMPTY
from gamemap import ReversiMap
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
from reversi_config import ReversiCheckPointConfig, ReversiGeneralConfig
from read_wth import read_wthor_files
from device_detect import detectDevice


def df_to_tensor(data: pd.DataFrame, device):
    dataSize = len(data)
    datalist = []
    for i in range(dataSize):
        datalist.append(data.iloc[i, 0])
        datalist.append(data.iloc[i, 1])
    datalist = torch.tensor(datalist, device=device)
    data = datalist.reshape((dataSize, 2, 8, 8))
    return data


def loadNetOutput(training_step: int, data: List[pd.DataFrame], device):
    with open(f"./models/preNet_{training_step}00.pt", 'rb') as f:
        model: "ResNet" = torch.load(f, map_location=device)
    datalist = load_dataframe(data, 64)
    batch_list = []
    for data in datalist:
        layer_out_list = []  # 储存一代神经网络中15个残差块的输出
        layer_out = model.initConv(df_to_tensor(data, device))
        for layer in model.layers:
            layer_out = layer(layer_out)
            layer_out_list.append(layer_out)
        batch_list.append(layer_out_list)
    trans_list = [[batch_list[k][i] for k in range(
        len(batch_list))] for i in range(len(model.layers))]
    result = []
    for lst in trans_list:
        new_lst = lst[0]
        for ls in lst[1:]:
            new_lst = torch.cat([new_lst, ls], dim=0)
        result.append(new_lst)
    return result


if __name__ == '__main__':
    globalDevice = detectDevice()

    paths = ['./gamedata/WTH_' + str(i)+'.wtb' for i in range(1977, 1981)]
    trainset, testset = read_wthor_files(paths)

    for training_step in range(1, 58):
        loadNetOutput(training_step, trainset, globalDevice)
