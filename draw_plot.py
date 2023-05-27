import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import math

# 画图可以参考：
# https://blog.csdn.net/AnimateX/article/details/122016419
# https://blog.csdn.net/u013185349/article/details/122618862


def draw_3d(concept_name: str, acc_list):

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    # 绘制散点图
    layer_list = np.arange(0, 15)
    epoch_list = np.arange(100, 5800, 100)
    w, b = np.meshgrid(layer_list, epoch_list)
    # 确认一下acc_list的x,y维度对不对，只要打印出size看一下就可以了
    surf = ax.plot_surface(w, b, acc_list, rstride=1,
                           cstride=1, cmap=cm.coolwarm)

    # ax.plot(layer_list, epoch_list, acc_list)

    # 设置坐标轴的标签
    ax.set_xlabel('Block')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('Test Accuracy')

    ax.set_title(f'Concept:{concept_name}')

    # 设置坐标轴的取值范围
    ax.set_xlim([15, 0])
    ax.set_ylim([0, 6000])
    ax.set_zlim([0, 1])

    # 保存图片
    plt.savefig(f'./plots/{concept_name}.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    prefix_0 = 'concept: '
    prefix_1 = 'accuracy_step: '
    prefix_2 = 'accuracy_step_0: '
    with open('./log_file/log_acc.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 5):
        concept_name = lines[i][len(prefix_0):-1]
        accuracy_step = np.array(eval(lines[i+2][len(prefix_2):]))
        draw_3d(concept_name, accuracy_step)
