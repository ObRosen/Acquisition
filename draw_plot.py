import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# 画图可以参考这个：
# https://blog.csdn.net/AnimateX/article/details/122016419
# https://blog.csdn.net/u013185349/article/details/122618862


def draw_3d(concept_name: str, acc_list):

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    # 绘制散点图
    layer_list = np.arange(0, 15)
    epoch_list = np.arange(1, 58)
    w, b = np.meshgrid(layer_list, epoch_list)
    # 确认一下acc_list的x,y维度对不对，只要打印出size看一下就可以了
    ax.plot_surface(w, b, acc_list, cmap=cm.coolwarm)

    # ax.plot(layer_list, epoch_list, acc_list)

    # 设置坐标轴的标签
    ax.set_xlabel('Block')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('Test Accuracy')

    ax.set_title(f'Concept:{concept_name}')

    # 设置坐标轴的取值范围
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 60])
    ax.set_zlim([-10.0, 1.0])

    # 保存图片
    plt.savefig(f'./plots/{concept_name}.png')


