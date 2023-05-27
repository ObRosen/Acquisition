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
from rewrite_print import print, print_acc
import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告


batchSize = 128


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
    # l2_loss = torch.norm(w)/pred.size(0)
    l1_loss = torch.sum(torch.abs(w))/pred.size(0)
    return mse_loss + lamb * l1_loss  # 返回的是平均loss


# 定义训练函数
def train(model: LinearModel, X_train: torch.Tensor, y_train: torch.Tensor, lamb: float, device: torch.device, learning_rate, num_epochs=500):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    schduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=30, factor=0.7)

    X_train = load_tensor(X_train.clone().detach(), batchSize)
    y_train = load_tensor(y_train.reshape(-1), batchSize)

    for epoch in range(num_epochs):
        total_loss = 0
        datasize = 0
        for i in range(len(X_train)):

            inputs = X_train[i]
            targets = y_train[i]
            datasize += targets.size(0)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_f1(outputs, targets, model.linear.weight, lamb, device)
            total_loss += loss.item()*targets.size(0)
            loss.backward()
            optimizer.step()
            schduler.step(loss)

        if num_epochs > 500 and (epoch+1) % 500 == 0:
            print('LR: {}, Epoch [{}/{}], Loss: {:.4f}'.format(
                learning_rate, epoch + 1, num_epochs, total_loss/datasize))

# 正则化参数通过交叉验证确定


def decide_lambda_lr(z_d: torch.Tensor, c_z: torch.Tensor, n_splits: int, device: torch.device, random_seed: int = 42):  # z_d为probe的输入，c_z为标签数据
    kf = KFold(n_splits, shuffle=True, random_state=random_seed)
    lamb_list = [0.01, 0.1, 1, 10]  # 网格化搜索之后发现lamb的影响很小，为节省时间干脆都定为10
    lr_list = [1e-06, 1e-07, 1e-08, 1e-09]
    best_loss_lamb = float('inf')
    best_lamb = None
    best_lamb_lr = None

    for lamb in lamb_list:  # 决定了best_lamb及其下的best_loss和best_lr
        best_lr = None
        best_loss = float('inf')
        for lr in lr_list:  # 决定了当前lamb下的best_loss和best_lr
            total_loss = 0
            datasize = 0
            for k, index in enumerate(kf.split(z_d)):
                train_index, val_index = index
                X_train, X_val = z_d[train_index], z_d[val_index]
                y_train, y_val = c_z[train_index], c_z[val_index]
                # 定义模型并训练
                input_size = z_d.shape[1]*z_d.shape[2]*z_d.shape[3]
                linear_model = LinearModel(input_size)
                linear_model = linear_model.to(device)
                # print(f'-------------The {k+1}/{n_splits} th training------------')
                train(linear_model, X_train.reshape(-1, input_size),
                      y_train.reshape(-1, 1), lamb, device, learning_rate=lr, num_epochs=500)

                # 验证
                inputs = load_tensor(X_val)
                targets = torch.tensor(y_val, device=device)
                batch_output = linear_model(inputs[0])
                for i in range(1, len(inputs)):
                    batch_output = torch.cat(
                        [batch_output, linear_model(inputs[i])], dim=0)

                loss = loss_f1(batch_output, targets,
                               linear_model.linear.weight, lamb, device)
                total_loss += loss.item()*targets.size(0)
                datasize += targets.size(0)

            avg_loss = total_loss / datasize

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_lr = lr

        print(f"lambda: {lamb}, best lr: {best_lr}, best loss: {best_loss}")

        if best_loss < best_loss_lamb:
            best_loss_lamb = best_loss
            best_lamb = lamb
            best_lamb_lr = best_lr

    print(
        f"best lambda: {best_lamb}, best lr: {best_lamb_lr}, best loss: {best_loss_lamb}")
    return best_lamb, best_lamb_lr


def concept_probe(concept_name: str, train_val_data: pd.DataFrame, train_zd: torch.Tensor, device: torch.device, num_epochs: int = 1000):
    y_train = []
    for i in range(len(train_val_data)):
        board = train_val_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_train.append(c_z)

    y_train = torch.tensor(y_train, device=device)

    best_lamb, best_lr = decide_lambda_lr(train_zd, y_train, 3, device)
    input_size = train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3]

    linear_model = LinearModel(input_size)
    linear_model = linear_model.to(device)
    train(linear_model, train_zd.reshape(-1, input_size),
          y_train.reshape(-1, 1), best_lamb, device, best_lr, num_epochs)
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

    test_zd = load_tensor(test_zd)
    X_test = model(test_zd[0])
    for i in range(1, len(test_zd)):
        X_test = torch.cat([X_test, model(test_zd[i])], dim=0)

    r_squared = r2_score(np.array(y_test), np.array(X_test.tolist()))
    r_squared_0 = r2_score_0(np.array(y_test), np.array(X_test.tolist()))
    print(f"r^2: {r_squared}, r^2_0: {r_squared_0}")
    print("---------------------------------------------------------------\n")
    return r_squared, r_squared_0


# 这个部分只算一次，之后从结果中拿取即可
def zdlist(device: torch.device, epoch_list: List[int], trainset, testset):
    all_trainzd_list = []
    all_testzd_list = []
    for training_step in epoch_list:
        train_z_d_list = loadNetOutput(
            training_step, into_input_format_2(trainset), device)
        test_z_d_list = loadNetOutput(
            training_step, into_input_format_2(testset), device)
        all_trainzd_list.append(train_z_d_list)
        all_testzd_list.append(test_z_d_list)
    return all_trainzd_list, all_testzd_list  # list的长度就等于epoch_list的长度


def compute_results(device: torch.device, training_step, concept, trainset, testset, train_z_d_list, test_z_d_list, layer_num=15):
    accuracy_layer = []
    accuracy_layer_0 = []

    for layer in range(layer_num):
        print(
            f"---------NN training epoch: {training_step}, Layer num: {layer+1}/15---------")
        linear_model = concept_probe(
            concept, trainset, train_z_d_list[layer], device)
        accuracy, accuracy_0 = test_accuracy(
            concept, linear_model, testset, test_z_d_list[layer])

        accuracy_layer.append(accuracy)  # 记录当前training_step下遍历各layers的accuracy
        accuracy_layer_0.append(accuracy_0)

    return accuracy_layer, accuracy_layer_0


if __name__ == '__main__':

    globalDevice = detectDevice()
    BLACK = -1

    file_list = range(2000, 2022)  # 代表数据年份，可选：range(1977,2024)
    epoch_list = range(1, 58)  # 代表神经网络训练代数，可选：range(1,58)
    concept_list = [frontier, steady, corner, mobility, frontier_diff,
                    score_f1, score, steady_diff,  corner_diff, mobility_diff]

    paths = ['./gamedata/WTH_' + str(i)+'.wtb' for i in file_list]
    trainset, testset = read_wthor_files(paths)

    all_trainzd_list, all_testzd_list = zdlist(
        globalDevice, epoch_list, trainset, testset)

    for concept in concept_list:
        print(
            f'Current time: {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
        print(
            f'Use datasets in years: {file_list[0]}~{file_list[-1]}. Size of trainset: {len(trainset)}')
        print(f'Concept: {str(concept.__name__)}\n')

        accuracy_step, accuracy_step_0 = [], []
        for training_step in epoch_list:
            train_z_d_list = all_trainzd_list[training_step]
            test_z_d_list = all_testzd_list[training_step]
            accuracy_layer, accuracy_layer_0 = compute_results(
                globalDevice, training_step, concept, trainset, testset, train_z_d_list, test_z_d_list)
            accuracy_step.append(accuracy_layer)
            accuracy_step_0.append(accuracy_layer_0)

        print_acc(f"concept: {str(concept.__name__)}")
        print_acc(f"accuracy_step: {accuracy_step}")
        print_acc(f"accuracy_step_0: {accuracy_step_0}")
        print_acc(
            '---------------------------------------------------------------\n')
