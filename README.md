# 代码使用

## 棋局数据获取、读取与处理

- 从[La base WTHOR | Fédération Française d'Othello (ffothello.org)](https://www.ffothello.org/informatique/la-base-wthor/)中可获取1979-2023年黑白棋锦标赛的.wtb格式数据文件，创建`gamedata`文件夹，将数据文件存入该文件夹。
- `read_wth.py`中定义了读取.wtb格式数据文件的函数`read_wthor_files()`，以路径为自变量，输出训练集和测试集。
- `load_data.py`中定义了整个程序中所需的各种数据格式处理的函数。有些最终没有用到，可以忽略。
	- `dedup()`用于数据集旋转对称去重。
	- `load_tensor()`与`load_dataframe()`将两种相应格式的数据集分batch，用于后续使用。

## 神经网络输出数据的获取

- `load_net.py`定义了`loadNetOutput()`函数，顾名思义，从models文件夹中的checkpoint文件中读取神经网络的结构，并输入此前生成的棋局数据集，获取相应每一层残差块的输出。

## 人类概念的封装

- `concept.py`中定义了各种策略的函数，参考了[OrangeX4](https://orangex4.cool/post/reversi/)的代码，在此基础上延伸了一些。

## 探针训练、测试评估

- `linear_probe.py`为本项目的主文件，在models文件夹和gamedata文件夹准备好的情况下，更改好需要的`file_list, epoch_list, concept_list`后直接运行此文件即可。
	- 本文件关于正确率试用了两种定义，源头是`r2_score()`和`r2_score_0()`，由此产生结果对`accuracy`与`accuracy_0`等，如不需要可以删去。
- 注意此文件中使用的`print()`函数在`rewrite_print.py`中重载过，使输出结果同时打印在屏幕和`log_file`文件夹中的日志文件里。

### linear_probe.py

定义了`LinearModel`类，由一个`Linear`层组成。一个损失函数`loss_f1`，由平方误差和一个$L_1$正则项(**稀疏探针要用L1正则**)组成。

- **`train()`函数**：输入模型、训练数据集、训练标签集、$\lambda$值、学习率、epoch值，`train()`函数完成了一个根据损失函数更新模型参数的过程。

- **`decide_lambda_lr()`函数**：输入模型、自变量、因变量、交叉验证折数，`decide_lambda()`函数将数据分为k折，依次取其中一个作为验证集，剩下的作为训练集，在训练集上对不同的$\lambda$取值和学习率取值进行遍历训练，然后在验证集上计算平均损失水平，选定平均损失水平最低的$\lambda$和学习率为最优$\lambda$与最优学习率，由此确定了最佳正则化系数和最佳学习率。

- **`concept_probe()`函数**：输入概念的名字、自变量训练集、神经网络输入训练集，`concept_probe()`函数首先生成一个与概念对应的因变量训练集$c(z^0)$，然后在自变量训练集上用上述方法训练出对于当前概念的最佳探针$g$，并返回这个线性模型。

- **`test_accuracy()`函数**：输入概念的名字、模型、自变量测试集、神经网络输入测试集，首先生成一个与概念对应的因变量测试集$c(z^0)$，然后在自变量测试集上测试模型的正确率(即$R^2$值)，并返回这个正确率。

- **`compute_results()`函数**：组合上述函数进行完整的计算。

## 三维曲面图绘制

- `draw_plot.py`从`log_file`中的正确率日志文件读取数据并画图。
- 内存方面的问题。

## 一些没做的补丁

- 许多参数写进config.ini通过reversi_config.py读取会更好，由于时间限制目前没有作这个处理。
- ……