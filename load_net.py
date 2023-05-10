import torch
import sys
from resnet import ResNet
from load_data import data_preprocess, data_loader
import pandas as pd
from player_color import BLACK, WHITE, EMPTY
from gamemap import ReversiMap
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
from reversi_config import ReversiCheckPointConfig, ReversiGeneralConfig
from read_wth import read_wthor_files
from device_detect import detectDevice


def df_to_tensor(data: pd.DataFrame):
    dataSize = len(data)
    datalist = []
    for i in range(dataSize):
        datalist.append(data.iloc[i, 0])
        datalist.append(data.iloc[i, 1])
    datalist = torch.tensor(datalist, device=globalDevice)
    data = datalist.reshape((dataSize, 2, 8, 8))
    return data


def loadNetOutput(training_step: int, data: pd.DataFrame, device):
    with open(f"./models/preNet_{training_step}00.pt", 'rb') as f:
        model: "ResNet" = torch.load(f, map_location=device)
    layer_out_list = []  # 储存一代神经网络中15个残差块的输出
    layer_out = model.initConv(df_to_tensor(data))
    for layer in model.layers:
        layer_out = layer(layer_out)
        layer_out_list.append(layer_out)
    return layer_out_list


if __name__ == '__main__':
    globalDevice = detectDevice()

    paths = ['.\gamedata\WTH_' + str(i)+'.wtb' for i in range(1977, 1981)]
    trainset, testset = read_wthor_files(paths)

    for training_step in range(1, 58):
        loadNetOutput(training_step, trainset, globalDevice)
