import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 画图可以参考这个：https://blog.csdn.net/AnimateX/article/details/122016419

# TODO: 从csv中读取数据，包括block、epoch和test accuracy等信息，存储在对应的列表中
block_list = [0,5,10]
epoch_list = [10,20,30]
acc_list = [1,2,3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.plot(block_list, epoch_list, acc_list)

# 设置坐标轴的标签
ax.set_xlabel('Block')
ax.set_ylabel('Epoch')
ax.set_zlabel('Test Accuracy')
ax.set_title('Concept')

# 设置坐标轴的取值范围
ax.set_xlim([0, 15])
ax.set_ylim([0, 30])
ax.set_zlim([0.0, 10.0])

plt.show()
# 保存图片
#plt.savefig('path/to/save/image.png')


"""
在上述代码中，首先要准备好数据，包括block、epoch和test accuracy等信息，
分别存储在对应的列表中。然后创建一个figure对象和一个Axes3D对象，用来绘制
三维图像。使用scatter()方法绘制散点图，其中block_list、epoch_list和
acc_list分别表示x轴、y轴和z轴上的取值。接着使用set_xlabel()、set_ylabel()
和set_zlabel()方法设置坐标轴的标签，使用set_xlim()、set_ylim()和set_zlim()
方法设置坐标轴的取值范围。最后使用savefig()方法将生成的图片保存在指定的文
件路径中。
"""
