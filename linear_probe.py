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
from draw_plot import draw_3d


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


# 定义损失函数和L1正则项 #或者改成L2正则项？
def loss_fn(pred: torch.Tensor, target: torch.Tensor, w, lamb):
    mse_loss = torch.mean((pred - target)**2)
    l1_loss = torch.sum(torch.abs(w))
    total_loss = mse_loss + lamb * l1_loss
    return torch.Tensor(total_loss)


# 定义训练函数
def train(model: LinearModel, X_train, y_train, lamb, learning_rate=0.01, num_epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs = X_train
    targets = y_train
    for epoch in range(num_epochs):
        outputs = model(inputs)
        optimizer.zero_grad()
        # 计算损失并更新权重
        loss = loss_fn(outputs, targets, model.linear.weight, lamb)
        # print(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()

        if num_epochs == 1000 and (epoch+1) % 100 == 0:
            print('Lambda: {}, Epoch [{}/{}], Loss: {:.4f}'.format(
                lamb, epoch + 1, num_epochs, loss.item()))


# L1正则化参数通过交叉验证确定
def decide_lambda(z_d: torch.Tensor, c_z: torch.Tensor, n_splits: int, random_seed: int = 42):  # z_d为probe的输入，c_z为标签数据
    kf = KFold(n_splits, shuffle=True, random_state=random_seed)
    lamb_list = [0.001, 0.01, 0.1, 1]
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
            # print(f'-------------The {k+1}/{n_splits} th training------------')
            train(linear_model, X_train.reshape(-1, z_d.shape[1]*z_d.shape[2]*z_d.shape[3]),
                  y_train.reshape(-1, 1), lamb, num_epochs=500)

            # 验证
            inputs = X_val
            targets = torch.Tensor(y_val)
            outputs = linear_model(inputs)
            loss = loss_fn(outputs, targets, linear_model.linear.weight, lamb)
            total_loss += loss.item()

        avg_loss = total_loss / n_splits
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

    # train_zd=batch_normalize(train_zd)
    # y_train = batch_normalize(y_train)  # 正则化用在什么地方合理？参数应该如何设置？
    y_train = torch.Tensor(y_train)

    best_lamb = decide_lambda(train_zd, y_train, 5)
    # best_lamb=0.1
    linear_model = LinearModel(
        train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3])
    train(linear_model, train_zd.reshape(-1,
                                         train_zd.shape[1]*train_zd.shape[2]*train_zd.shape[3]), y_train.reshape(-1, 1), best_lamb, num_epochs=1000)
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

    r_squared = r2_score(y_test, X_test.detach().numpy())
    # print("r^2: ", r_squared)
    return r_squared


def compute_results(file_list, epoch_list, concept_list, layer_num=15):
    paths = ['.\gamedata\WTH_' + str(i)+'.wtb' for i in file_list]
    trainset, testset = read_wthor_files(paths)

    accuracy_step = []
    for training_step in epoch_list:
        accuracy_concept = []
        train_z_d_list = loadNetOutput(
            training_step, into_input_format_2(trainset), globalDevice)
        test_z_d_list = loadNetOutput(
            training_step, into_input_format_2(testset), globalDevice)

        for concept in concept_list:
            accuracy_concept_layer = []

            for layer in range(layer_num):
                linear_model = concept_probe(
                    concept, trainset, train_z_d_list[layer])
                accuracy = test_accuracy(
                    concept, linear_model, testset, test_z_d_list[layer])
                print(accuracy)
                accuracy_concept_layer.append(accuracy)
            accuracy_concept.append(accuracy_concept_layer)
        accuracy_step.append(accuracy_concept)
    return accuracy_step


def make_plots(concept_list, accuracy_step):
    for idx, concept_name in enumerate(concept_list):
        acc_list = []
        for item in accuracy_step:
            acc_list.append(item[idx])
        draw_3d(concept_name, acc_list)


if __name__ == '__main__':
    globalDevice = detectDevice()
    BLACK = -1

    file_list = range(1977, 1981)  # range(1977,2024)
    epoch_list = range(1, 2)  # range(1,58)
    concept_list = [mobility]

    # , mobility_diff , frontier, frontier_diff, steady, corner, corner_steady, steady_diff, corner_diff, score_f1, score]

    accuracy_step = compute_results(file_list, epoch_list, concept_list)
    with open('acc.txt', 'w', encoding='UTF-8') as f:
        print(accuracy_step, file=f)
    # make_plots(concept_list, accuracy_step)
