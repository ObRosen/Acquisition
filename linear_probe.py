import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from concept import mobility, mobility_diff, frontier, frontier_diff, steady, corner, corner_steady, steady_diff, corner_diff, score_f1, score
from load_data import data_loader
import pandas as pd
from player_color import BLACK, WHITE, EMPTY

# 定义线性模型g


class LinearModel(nn.Module):  # 用于c-con
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 8*4*4

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.linear(x)
        return y.squeeze()

# 定义逻辑回归模型


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        return out


# 定义损失函数和L1正则项
def loss_fn(pred, target, w, lamb):
    mse_loss = torch.mean((pred - target)**2)
    l1_loss = torch.sum(torch.abs(w))
    total_loss = mse_loss + lamb * l1_loss
    return total_loss


# 定义训练函数
def train(model: LinearModel, X_train, y_train, lamb, learning_rate=0.01, num_epochs=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        # inputs = torch.autograd.Variable(torch.from_numpy(X_train)).float()
        # targets = torch.autograd.Variable(torch.from_numpy(y_train)).float()
        inputs = X_train
        targets = torch.Tensor(y_train)

        optimizer.zero_grad()

        # 计算预测值
        outputs = model(inputs)

        # 计算损失并更新权重
        loss = loss_fn(outputs, targets, model.linear.weight, lamb)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print('Lambda: {}, Epoch [{}/{}], Loss: {:.4f}'.format(lamb, epoch +
                  1, num_epochs, loss.item()))


# L1正则化参数通过交叉验证确定
def decide_lambda(model: LinearModel, z_d: torch.Tensor, c_z: torch.Tensor, n_splits: int, random_seed: int = 42):  # z_d为probe的输入，c_z为标签数据
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
            linear_model = model(z_d.shape[1]*z_d.shape[2]*z_d.shape[3])
            print(
                '-------------The {}/{} th training------------'.format(k+1, kf.get_n_splits))
            train(linear_model, X_train.reshape(-1, z_d.shape[1]*z_d.shape[2]*z_d.shape[3]),
                  y_train.reshape(-1, 1), lamb, num_epochs=500)

            # 验证
            inputs = X_val
            targets = torch.Tensor(y_val)
            outputs = linear_model(inputs)
            loss = loss_fn(outputs, targets, linear_model.linear.weight, lamb)
            total_loss += loss.item()

        avg_loss = total_loss / kf.get_n_splits()
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lamb = lamb

    print("best lambda:", best_lamb)
    return best_lamb


def concept_probe(concept_name: str, train_val_data: pd.DataFrame, train_zd: torch.Tensor):
    y_train = []

    for i in range(len(train_val_data)):
        board = train_val_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_train.append(c_z)
    print(y_train)

    linear_model = LinearModel(
        train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3])
    best_lamb = decide_lambda(linear_model, train_zd, y_train, 5)
    train(linear_model, train_zd.reshape(-1,
                                         train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3]), y_train.reshape(-1, 1), best_lamb, num_epochs=1000)
    return linear_model


def test_accuracy(concept_name: str, model: LinearModel, test_data: pd.DataFrame, test_z_d: torch.Tensor):
    y_test = []
    for i in range(len(test_data)):
        board = test_data.iloc[i, :]
        c_z = concept_name(board, BLACK)
        y_test.append(c_z)
    print(y_test)

    X_test = model.forward(test_z_d)

    r_squared = r2_score(y_test, X_test)
    print("r^2: ", r_squared)
    return r_squared


if __name__ == '__main__':
    # TODO:数据准备,需要从load_data中获取神经网络的输出
    train_z_d = []
    test_z_d = []
    BLACK = -1
    # 在测试集上测试
    data = pd.read_csv('./Acquisition/allgamedata_small.csv', encoding='utf-8')
    trainset, valset, testset = data_loader(data, 15)
    linear_model = concept_probe(mobility_diff, trainset, train_z_d)
    accuracy = test_accuracy(mobility_diff, linear_model, testset, test_z_d)
    # TODO:输入三个list供draw_plot.py使用