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


def detectDevice() -> torch.device:
    if not torch.cuda.is_available():
        print("cuda device not available, use cpu")
        return torch.device("cpu")

    print("searching for free gpu...")

    # set cuda device
    if sys.platform == "win32":
        print("windows detected, use cuda:0")
        deviceIndex = 0  # not supporting detect free gpu on windows
    else:
        deviceIndex = queryFreeGPU()
        deviceIndex = int(deviceIndex)
        print("use device: cuda:", deviceIndex, "(GPU)")

    device = torch.device("cuda", deviceIndex)
    return device


def df_to_tensor(data: pd.DataFrame):
    dataSize = len(data)
    datalist = []
    for i in range(dataSize):
        datalist.append(data.iloc[i, 0])
        datalist.append(data.iloc[i, 1])
    datalist = torch.tensor(datalist, device=globalDevice)
    data = datalist.reshape((dataSize, 2, 8, 8))
    return data

if __name__=='__main__':
    globalDevice = detectDevice()

    with open(f"./models/preNet_100.pt", 'rb') as f:
        model: "ResNet" = torch.load(f, map_location=globalDevice)
    
    paths = ['.\gamedata\WTH_' +
             str(i)+'.wtb' for i in range(1977, 1981)]  # range(1977,2024)
    dataset = read_wthor_files(paths)
    trainset, testset = data_loader(dataset, 15)
    # batchStateInput: List[List[List[List[float]]]], size: (dataSize, 2, 8, 8)
    
    layer_out_list = [] # 储存一代神经网络中15个残差块的输出
    layer_out = model.initConv(df_to_tensor(trainset))
    for layer in model.layers:
        layer_out=layer(layer_out)
        layer_out_list.append(layer_out)

    
    

