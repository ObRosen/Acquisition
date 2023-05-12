import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from kfold import KFold
from concept import mobility, mobility_diff, frontier, frontier_diff, steady, corner, corner_steady, steady_diff, corner_diff, score_f1, score
from read_wth import read_wthor_files
import pandas as pd
from player_color import BLACK, WHITE, EMPTY
from device_detect import detectDevice
from load_net import loadNetOutput
from load_data import into_input_format_2, load_tensor
import time
import random


class LinearModel(nn.Module):  # 定义线性模型g,用于c-con
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        y = self.linear(x)
        return y.squeeze()


def batch_normalize(x, gamma=0.5, beta=0.5, eps=1e-5):
    # N, D = x.shape
    mean = np.mean(x)
    var = np.var(x)
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    return out


def loss_f1(pred: torch.Tensor, target: torch.Tensor, w, lamb, device: torch.device) -> torch.Tensor:
    mse_loss = torch.mean((pred - target)**2)
    l1_loss = torch.sum(torch.abs(w))/pred.size(0)
    # print(mse_loss)
    # print(l1_loss)
    return mse_loss + lamb * l1_loss


def loss_f2(pred: torch.Tensor, target: torch.Tensor, w, lamb, device: torch.device) -> torch.Tensor:
    mse_loss = torch.mean((pred - target)**2)
    l2_loss = torch.norm(w)/pred.size(0)
    return mse_loss + lamb * l2_loss


def criterion(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> torch.Tensor:
    mse_loss = torch.mean((pred - target)**2)
    # print(mse_loss)
    # l2_loss = torch.norm(w)/pred.size(0)
    # print(l2_loss)
    return mse_loss


# 定义训练函数
def train(model: LinearModel, X_train: torch.Tensor, y_train: torch.Tensor, lamb: float, device: torch.device, learning_rate, num_epochs=500):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    schduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=30, factor=0.7)

    X_train = load_tensor(X_train.clone().detach(), 128)
    y_train = load_tensor(y_train, 128)

    for epoch in range(num_epochs):
        for i in range(len(X_train)):

            inputs = X_train[i]
            targets = y_train[i]

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_f1(outputs, targets, model.linear.weight, lamb, device)
            # print(loss.item())
            loss.backward()
            optimizer.step()
            schduler.step(loss)

            if (epoch+1) % 100 == 0:
                print('LR: {}, Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                    learning_rate, epoch + 1, num_epochs, i+1, len(X_train), loss.item()))


# 正则化参数通过交叉验证确定
def decide_lambda(z_d: torch.Tensor, c_z: torch.Tensor, n_splits: int, device: torch.device, random_seed: int = 42):  # z_d为probe的输入，c_z为标签数据
    kf = KFold(n_splits, shuffle=True, random_state=random_seed)
    lamb_list = [0.001, 0.01, 0.1, 1, 10]
    lr_list = [1e-4, 1e-5, 1e-6, 1e-7]
    best_loss = float('inf')
    lamb = 0.1
    best_lr = None

    for lr in lr_list:
        total_loss = 0

        for k, index in enumerate(kf.split(z_d)):
            train_index, val_index = index
            X_train, X_val = z_d[train_index], z_d[val_index]
            y_train, y_val = c_z[train_index], c_z[val_index]
            # 定义模型并训练
            input_size = z_d.shape[1]*z_d.shape[2]*z_d.shape[3]
            linear_model = LinearModel(input_size)
            linear_model = linear_model.to(device)
            print(f'-------------The {k+1}/{n_splits} th training------------')
            train(linear_model, X_train.reshape(-1, input_size),
                  y_train.reshape(-1, 1), lamb, device, learning_rate=lr, num_epochs=500)

            # 验证
            inputs = X_val
            targets = torch.tensor(y_val, device=device)
            # targets=y_val.clone().detach()
            outputs = linear_model(inputs)
            loss = loss_f1(outputs, targets,
                           linear_model.linear.weight, lamb, device)
            total_loss += loss.item()

        avg_loss = total_loss / kf.get_n_splits()
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lr = lr

    print("best lr:", best_lr)
    return best_lr


def concept_probe(concept_name: str, train_val_data: pd.DataFrame, train_zd: torch.Tensor, device: torch.device):
    y_train = []
    for i in range(len(train_val_data)):
        board = train_val_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_train.append(c_z)

    y_train = torch.tensor(y_train, device=device)
    # print(y_train)

    # y_train = batch_normalize(y_train)  # 正则化用在什么地方合理？参数应该如何设置？

    best_lr = decide_lambda(train_zd, y_train, 5, device)
    best_lamb = 0.1
    input_size = train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3]

    linear_model = LinearModel(input_size)
    linear_model = linear_model.to(device)
    train(linear_model, train_zd.reshape(-1, input_size),
          y_train.reshape(-1, 1), best_lamb, device, best_lr, num_epochs=1000)
    return linear_model


def r2_score_0(y_test, X_test):
    SStot = np.sum((y_test-np.mean(y_test))**2)
    SSres = np.sum((y_test-X_test)**2)
    r2 = 1-SSres/SStot
    return r2


def r2_score(y_test, X_test):
    r = np.corrcoef(X_test, y_test)[0, 1]
    return r**2


def test_accuracy(concept_name: str, model: LinearModel, test_data: pd.DataFrame, test_zd: torch.Tensor):
    y_test = []
    for i in range(len(test_data)):
        board = test_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_test.append(c_z)

    # y_test = batch_normalize(y_test) # 该不该正则化？

    X_test = model(test_zd)
    r_squared = r2_score(np.array(y_test), np.array(X_test.tolist()))
    print("r^2: ", r_squared)
    return r_squared


def compute_results(device: torch.device, file_list, epoch_list, concept, layer_num=15):
    batchSize = 128
    paths = ['./gamedata/WTH_' + str(i)+'.wtb' for i in file_list]
    trainset, testset = read_wthor_files(paths)

    accuracy_step = []
    for training_step in epoch_list:
        accuracy_layer = []
        train_z_d_list = loadNetOutput(
            training_step, into_input_format_2(trainset), device)
        test_z_d_list = loadNetOutput(
            training_step, into_input_format_2(testset), device)

        for layer in range(layer_num):
            linear_model = concept_probe(
                concept, trainset, train_z_d_list[layer], device)
            accuracy = test_accuracy(
                concept, linear_model, testset, test_z_d_list[layer])
            # print(accuracy)
            accuracy_layer.append(accuracy)

        accuracy_step.append(accuracy_layer)
    return accuracy_step


if __name__ == '__main__':
    globalDevice = detectDevice()
    BLACK = -1

    file_list = range(1977, 1982)  # range(1977,2024)
    epoch_list = range(1, 2)  # range(1,58)
    concept = score_f1
    # , mobility_diff , frontier, frontier_diff, steady, corner, corner_steady, steady_diff, corner_diff, score_f1, score]

    accuracy_step = compute_results(
        globalDevice, file_list, epoch_list, concept)
    with open('acc.txt', 'w', encoding='UTF-8') as f:
        print(f"concept: {concept}", file=f)
        print(accuracy_step, file=f)
    # make_plots(concept_list, accuracy_step)
