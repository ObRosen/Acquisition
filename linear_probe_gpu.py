# gpu版本

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kfold import KFold
from concept import mobility, mobility_diff, frontier, frontier_diff, steady, corner, corner_steady, steady_diff, corner_diff, score_f1, score
from read_wth import read_wthor_files
import pandas as pd
from player_color import BLACK, WHITE, EMPTY
from device_detect import detectDevice
from load_net import loadNetOutput
from load_data import into_input_format_2


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


def trans(x):
    return torch.autograd.Variable(x.to(torch.float32), requires_grad=True).to(globalDevice)

# 定义损失函数和L1正则项 #或者改成L2正则项？


def loss_fn(pred: torch.Tensor, target: torch.Tensor, w, lamb, device: torch.device) -> torch.Tensor:
    mse_loss = torch.mean((pred - target)**2)
    l1_loss = torch.sum(torch.abs(w))/pred.size(0)
    total_loss = mse_loss + lamb * l1_loss
    return torch.tensor(total_loss, device=device)
    # return total_loss.clone().detach()


def criterion(pred: torch.Tensor, target: torch.Tensor, w, lamb, device: torch.device) -> torch.Tensor:
    mse_loss = torch.mean((trans(pred) - trans(target))**2)
    l2_loss = torch.tensor(torch.norm(trans(w))/pred.size(0), device=device)
    total_loss = trans(mse_loss + lamb * l2_loss)
    return total_loss.clone().detach().requires_grad_(True)


# 定义训练函数
def train(model: LinearModel, X_train: torch.Tensor, y_train: torch.Tensor, lamb: float, device: torch.device, learning_rate=0.01, num_epochs=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # inputs = trans(X_train)
    inputs = torch.tensor(X_train, dtype=torch.float32,
                          requires_grad=True, device=device)
    targets = trans(y_train)
    for epoch in range(num_epochs):
        outputs = model(inputs)
        optimizer.zero_grad()
        # 计算损失并更新权重
        # print([x.grad for x in optimizer.param_groups[0]['params']])
        loss = criterion(outputs, targets, model.linear.weight, lamb, device)
        loss.backward()
        optimizer.step()  # optimizer参数梯度为None，没有更新成功，可能是使用gpu加速的问题（在cpu机器上没有这个问题）

        # if (epoch+1) % 100 == 0:
        # print('Lambda: {}, Epoch [{}/{}], Loss: {:.4f}'.format(lamb, epoch + 1, num_epochs, loss.item()))


# 正则化参数通过交叉验证确定
def decide_lambda(z_d: torch.Tensor, c_z: torch.Tensor, n_splits: int, device: torch.device, random_seed: int = 42):  # z_d为probe的输入，c_z为标签数据
    kf = KFold(n_splits, shuffle=True, random_state=random_seed)
    lamb_list = [0.001, 0.01, 0.1, 1, 10]
    best_loss = float('inf')
    best_lamb = None

    for lamb in lamb_list:
        total_loss = 0

        for k, index in enumerate(kf.split(z_d)):
            train_index, val_index = index
            X_train, X_val = z_d[train_index], z_d[val_index]
            y_train, y_val = c_z[train_index], c_z[val_index]
            # 定义模型并训练
            linear_model = LinearModel(z_d.shape[1]*z_d.shape[2]*z_d.shape[3])
            linear_model.to(device)
            # print(f'-------------The {k+1}/{n_splits} th training------------')
            train(linear_model, X_train.reshape(-1, z_d.shape[1]*z_d.shape[2]*z_d.shape[3]),
                  y_train.reshape(-1, 1), lamb, device, num_epochs=500)

            # 验证
            inputs = X_val
            targets = torch.tensor(y_val, device=device)
            # targets=y_val.clone().detach()
            outputs = linear_model(inputs)
            loss = loss_fn(outputs, targets,
                           linear_model.linear.weight, lamb, device)
            total_loss += loss.item()

        avg_loss = total_loss / kf.get_n_splits()
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lamb = lamb

    print("best lambda:", best_lamb)
    return best_lamb


def concept_probe(concept_name: str, train_val_data: pd.DataFrame, train_zd: torch.Tensor, device: torch.device):
    y_train = []
    for i in range(len(train_val_data)):
        board = train_val_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_train.append(c_z)

    # train_zd=batch_normalize(train_zd)
    # y_train = batch_normalize(y_train)  # 正则化用在什么地方合理？参数应该如何设置？
    y_train = torch.tensor(y_train, device=device)

    best_lamb = decide_lambda(train_zd, y_train, 5, device)
    # best_lamb=0.1
    linear_model = LinearModel(
        train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3]).to(device)
    train(linear_model, train_zd.reshape(-1,
                                         train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3]), y_train.reshape(-1, 1), best_lamb, device, num_epochs=1000)
    return linear_model


def r2_score(y_test, X_test):
    SStot = np.sum((y_test-np.mean(y_test))**2)
    SSres = np.sum((y_test-X_test)**2)
    r2 = 1-SSres/SStot
    return r2


def test_accuracy(concept_name: str, model: LinearModel, test_data: pd.DataFrame, test_z_d: torch.Tensor):
    y_test = []
    for i in range(len(test_data)):
        board = test_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_test.append(c_z)

    # y_test = batch_normalize(y_test) # 该不该正则化？
    X_test = model(test_z_d)
    r_squared = r2_score(np.array(y_test), np.array(X_test.tolist()))
    print("r^2: ", r_squared)
    return r_squared


if __name__ == '__main__':
    globalDevice = detectDevice()
    BLACK = -1
    paths = ['./gamedata/WTH_' + str(i)+'.wtb' for i in range(1977, 1983)]
    trainset, testset = read_wthor_files(paths)
    # , mobility_diff, frontier, frontier_diff, steady, corner, corner_steady, steady_diff, corner_diff, score_f1, score]
    concept_list = [mobility]

    accuracy_step_concept = []
    for training_step in range(1, 2):
        accuracy_concept = []
        train_z_d_list = loadNetOutput(
            training_step, into_input_format_2(trainset), globalDevice)
        test_z_d_list = loadNetOutput(
            training_step, into_input_format_2(testset), globalDevice)
        for concept in concept_list:
            for layer in range(1):
                linear_model = concept_probe(
                    concept, trainset, train_z_d_list[layer], globalDevice)
                accuracy = test_accuracy(
                    concept, linear_model, testset, test_z_d_list[layer])
                # print(accuracy)
                accuracy_concept.append(accuracy)
        accuracy_step_concept.append(accuracy_concept)
