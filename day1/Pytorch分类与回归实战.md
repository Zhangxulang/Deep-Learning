 

# Pytorch分类与回归实战

### 官网介绍

要查看 PyTorch 官方文档和API ，请访问 PyTorch 官方网站（https://pytorch.org/）。在网站的顶部菜单中，您会找到一个名为 "Docs"  的选项。将鼠标悬停在上面，会出现一个下拉菜单，其中包 含各种文档资源。

您可以选择 "Stable"  版本或 "Latest"  版本，具体取决于您的需求和 PyTorch  的版本。单击所选版本后，将打开官方文档页面。

在官方文档页面中，您可以找到以下内容：

1. 教程（Tutorials）：提供了一系列教程，介绍了如何使用 PyTorch 进行各种任务，包括图像分类、 目标检测、 自然语言处理等。

2. API  文档（API Documentation）：这是官方提供的详尽的 API  文 档，涵盖了 PyTorch 中的各种类、函数和模块。您可以通过类别、 模块或关键字搜索来查找您感兴趣的内容。 

3. 示例（Examples）：提供了一些示例代码，展示了如何使用 PyTorch 进行常见任务的实现方法。

4. 高级主题（Advanced Topics）：讨论了一些高级主题，如自定义模型、分布式训练、模型部署等。

API  文档是软件开发中的重要资源，它提供了关于库、框架或工具中可用类、函数、方法和属性的详细信息。对于 PyTorch ，官方的 API 文档提供了关于 PyTorch  库中各个模块、类和函数的详细说明。

在 PyTorch  的 API  文档中，您可以找到以下内容：

1. 模块（Modules）：PyTorch  提供了许多模块，如 `torch`、`torch.nn`、 `torch.optim`  等。每个模块都有自己的文档页面，展示了该模块中定 义的类、函数和常量。您可以在这些页面上找到每个项的详细说明、 参数列表和示例代码。 
2. Classes：PyTorch中的类代表了各种对象如张量（torch.Tensor）、神经网络模型（torch.nn.Module）等。每个类都有自己的文档页面，其中包含了该类的构造函数、属性、方法和示例-----重点
3. 函数（Functions）：PyTorch提供了许多函数，用于执行各种操作，如数学运算、张量操作、梯度计算等。每个函数都有自己的文档页面，展示了函数的参数、返回值和使用示例。---重点

4. 常量（Constants）：PyTorch  定义了一些常用的常量，如损失函数、优化算法等。这些常量也有自己的文档页面，提供了常量的说明和用法示例。 

在 API  文档中，您可以通过浏览目录、使用搜索功能或按模块和类 别浏览来查找和查看感兴趣的内容。每个页面都包含了清晰的描述、 参数说明、返回值说明和使用示例，帮助您理解和使用 PyTorch 中 的不同功能。

请注意，阅读和理解API文档需要一定的熟悉度和背景知识。在使 用 PyTorch 进行开发时，参考官方 API  文档是学习和使用 PyTorch 的重要资源之一。

# 分类问题与回归问题

◆分类问题预测的是类别,模型的输出是概率分布

◆三分类问题输出例子: [0.2, 0.7, 0.1]

◆回归问题预测的是值,模型的输出是一个实数值

![img](file:///C:\Windows\Temp\ksohtml30060\wps1.png) 

为什么需要目标函数?

◆参数是逐步调整的

◆目标函数可以帮助衡量模型的好坏

◆ Model A: [0.1, 0.4, 0.5]

◆ Model B: [0.1, 0.2, 0.7]

softmax 就是一个目标函数 ：将结果变为一个概率值



# 分类问题

 

◆需要衡量目标类别与当前预测的差距

◆三分类问题输出例子: [0.2, 0.7, 0.1]

◆三分类真实类别: 2 -> one_ hot-> [0, 0, 1]

◆One-hot 编码,  把正整数变为向量表达

 

◆生成一个长度不小于正整数的向量,只有正整数的位置处为1 其余位置都为0

 

下面链接是 one-hot 编码解释

https://www.cnblogs.com/shuaishuaidefeizhu/p/11269257.html 我们可以预测值和真实值均转换为one-hot 计算距离

 

 

***\*Softmax\**** ***\*解析\****

 

[https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0](https://zh.wikipedia.org/wiki/Softmax函数)

![img](file:///C:\Windows\Temp\ksohtml30060\wps2.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps3.png) 

 

 

 

***\*交叉熵损失和均方误差损失对比分析\****

 

均方误差(回归）

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps4.png) 

***\*交叉熵损失\****

 

 

 





![img](file:///C:\Windows\Temp\ksohtml30060\wps5.png) 

结论：

 

.     相同点 ：当输出值与真实值接近的话 ，cross_entropy 和 rmse 的值都会接近 0

.     cross_entropy 具有 rmse 不具有的优点 ：***\*避免学习速率降低的情况\****。 (注意这种效 果仅限于输出层 ， 隐藏层的学习速率与其使用的激活函数密切相关。)

 

.     主要原因是逻辑回归配合 MSE 损失函数时，采用梯度下降法进行学习时，会出现模

型一开始训练时 ，学习速率非常慢的情况（[MSE 损失函数](https://zhuanlan.zhihu.com/p/35707643)）

.     ***\*均方损失\**** ***\*：假设误差是正态分布\**** ***\*，适用于线性的输出(如回归问题)\**** ***\*，特\*******\*点是对于与\**** ***\*真实结果差别越大\**** ***\*，则惩罚力度越大\**** ***\*，这并不适用于分类问题\****

.     ***\*交叉熵损失：假设误差是二值\*******\*分布\**** ***\*，可以视为预测概率分布和真实概率分布的相似\****

 

***\*程度。在分类问题中有良好的应用。\****

下面这个更清晰（上课通过下面链接例子来讲解） https://zhuanlan.zhihu.com/p/35709485

***\*交叉熵损失\**** ***\*更能捕捉到预测效果的差异\****

 

 

***\*w\**** ***\*的初始分布\****

 

***\*g\**** ***\*lorot\*******\*_\*******\*uniform\**** 是均匀分布

从 [-limit，limit] 中的均匀分布中抽取样本，其中  limit 是 sqrt(6 / (fan_in + fan_out))， fan_in 是权值张量中的输入单位的数量， fan_out 是权值张量中的输出 单位的数量。比如 784+300，就是 np.sqrt(6/1084)

 

在[概率论](https://baike.baidu.com/item/概率论/829122)和[统计学](https://baike.baidu.com/item/统计学/1175)中，均匀分布也叫矩形分布，它是对称概率分布，在相同长度间隔的分布 概率是等可能的。 均匀分布由两个参数 a 和b 定义，它们是数轴上的最小值和最大值，通 常缩写为 U（ a，b）。





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps6.png) 

下面链接大家可以自行看一下

[https://baike.baidu.com/item/%E5%9D%87%E5%8C%80%E5%88%86%E5%B8%83/954451](https://baike.baidu.com/item/均匀分布/954451)

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps7.png) 

https://zhuanlan.zhihu.com/p/514912456 目标函数

 

分类问题

◆平方差损失举例

◆预测值: [0.2, 0.7, 0.1]

◆真实值:[0,0,1]

◆损失函数值: [(0-0)^2 + (0.7-0)^2 + (0.1-1)^2]*0.5=0.65

 

 

 

`loss.backward()`  是  PyTorch 中用于计算梯度的方法 。 它是在进行反向传播

（back propagation）过程中的一个关键步骤 。 下面是  `loss.backward()`  方法的主要作用：

 

1. 根据计算图： PyTorch 中的计算图是由前向传播过程中的张量操作构建的 。 当调用

`loss.backward()`  时 ， 它会遵循计算图中的连接关系，从损失节点开始向后传播，计算每个 相关参数的梯度。

 

2. 梯度计算： `loss.backward()`  方法会根据链式法则自动计算每个参数的梯度 。 它会沿着计 算图反向传播梯度，将梯度值累积到每个参数的  `.grad`  属性中。

 

3. 梯度累积：如果在调用  `loss.backward()`  前进行了多次前向传播和损失计算，那么每次 调用  `loss.backward()`  时 ，梯度将被累积到参数的  `.grad`  属性中 。这通常用于在训练过程 中使用小批量样本进行梯度更新。





 

4. 参数更新 ：在计算完梯度后， 可以使用优化器（如  `torch.optim`  中的优化器） 来更新模 型的参数，将梯度信息应用于参数更新规则。

总而言之， `loss.backward()`  的作用是根据计算图和损失函数，计算模型参数的梯度 。这些 梯度可以用于更新模型参数， 以便在训练过程中最小化损失函数。

 

 

***\*回归问题\****

 

◆预测值与真实值的差距.

◆平方差损失

◆绝对值损失

 

 

◆模型的训练就是调整参数,使得目标函数逐渐变小的过程

 

 

***\*3\****  ***\*分类回归实战\****

 

◆搭建分类模型

◆回调函数

◆搭建回归模型

 

 

 

 

***\*01-02_classification_m\*******\*odel.ipynb\****

 

数据集在

data 文件夹下

 

 

 

l loss：训练集损失值

l accuracy:训练集准确率

l val_loss:测试集损失值

l val_accruacy:测试集准确率 l

l 以下 5 种情况可供参考：

l train loss 不断下降，test loss 不断下降，说明网络仍在学习 ;（最好的）





l train loss 不断下降，test loss 趋于不变，说明网络过拟合 ;（max pool 或者正则化）

l train loss 趋于不变，test loss 不断下降，说明数据集 100%有问题 ;（检查 dataset）

l train loss 趋于不变，test loss 趋于不变，说明学习遇到瓶颈，需要减小学习率或 批量(batch_size)数目 ;（减少学习率）

l train loss 不断上升，test loss 不断上升，说明网络结构设计不当，训练超参数设 置不当，数据集经过清洗等问题。

sparse_categorical_crossentropy 计算稀疏分类 crossentropy 损失。 categorical_crossentropy 计算分类 crossentropy 损失

如果你的 targets 是 one-hot 编码，用 categorical_crossentropy one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]

如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy 数字编码：2, 0, 1

 

 

***\*为什么要分为训练集，\**** ***\*验证集，\**** ***\*测试集\****

 

***\*为了防止人工调参造成的信息泄露\****

***\*详见\**** ***\*4.2.1\**** ***\*训练集、验证集和测试集\****

 

 

***\*归一化与标准化\****

 

transforms.Normalize(mean, std) 将 图 像 张 量 的每 个通道 进 行 归 一 化 。 这 里 的 mean 和std 是根据数据集的特性和需求来确定的

通过将 Normalize 变换与其他变换（如 ToTensor()）组合在一起，您可以在加载 数据集时自动应用归一化操作。这样，训练数据将在输入模型之前进行归一化处 理。

***\*transforms.ToTensor()\****  ***\*主要执行以下操作：\****

 

 

1  缩放像素值：将图像中每个通道的像素值从 [0, 255] 归一化到 [0.0, 1.0] 。这是通过将每个像素值除以

255  来实现的。

 

 

2  增加通道维度：对于单通道的灰度图像，它会增加一个通道维度，使其成为具有两个维度的张量，即 [channels, height, width] 。对于三通道的彩色图像，它将保持三个通道。

 

 

3  转换数据类型：将图像数据转换为浮点数类型（float32），因为 PyTorch  的神经网络层通常处理浮点数 数据。

 





 

 

 

 

标准化之前的数据

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps8.png) 

 

loss: 0.3565

 

accuracy: 0.8722

 

***\*mean\**** ***\*=\**** ***\*(0,)\****

***\*std\**** ***\*=\**** ***\*(1,)\****

 

***\*#\**** ***\*定义数据集的变换\****

***\*transform\**** ***\*= transforms.Compose([\****

***\*transforms.ToTensor(),\****

***\*transforms.Normalize(mean, std)\****

***\*])\****

 

***\*进行标准化后\****

***\*loss:\****     ***\*0.3494\****

***\*accuracy:\**** ***\*0.8766\****

 

 

***\*mean\**** ***\*=\**** ***\*(0.5,)\****

***\*std\**** ***\*=\**** ***\*(0.5,)\****

 

***\*#\**** ***\*定义数据集的变换\****

***\*transform\**** ***\*= transforms.Compose([\****

***\*王道码农训练营\*******\*-WWW.CSKAOYAN.COM\****



***\*transforms.ToTensor(),\****

***\*transforms.Normalize(mean, std)\****

***\*])\****

 

***\*进行标准化后\****

***\*loss:\****     ***\*0.3255\****

***\*accuracy:\**** ***\*0.8858\****

 

***\*通过标准化后发觉准确率提升！\****

 

这样做的 目 的是将数据的分布调整为均值为 0 ，标准差为 1 的分布 ，从而减少了不同 特征之间的尺度差异 ，有助于模型更快地收敛 ，并且对学习率的选择更为鲁棒。

 

 

 

 

***\*回调函数\****

 

03_classification_model_more_control.ipynb

 

这部分主要增加了  tensorboard_callback

SaveCheckpointsCallback EarlyStopCallback

 

 

***\*earlystopping\****

 

| ***\*min_delta\**** | ***\*监视数量的最小变化有资格作为改进，\**** ***\*即绝\**** ***\*对变化小于\**** ***\*min\*******\*_\*******\*delta\**** ***\*，将不视为改进。\**** |
| ------------------- | ------------------------------------------------------------ |
| ***\*patience\****  | ***\*没有改善的时期数\**** ***\*，之后训练将停止。\****      |

 

 

***\*Model\**** ***\*Checkpoint（保存模型）\****

 

***\*TensorBoard\****

 

 

TensorBoard

然后访问 http://localhost:8848/ 得到下图





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps9.png) 

Smoothing 是平滑， 可以调整为零

 

***\*Graph\****

 

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps10.png) 

 

 

***\*回归模型  03_regression\****

 

 

数据在家目录的 scikit_learn_data 下 ，数据如果下载不下来， 可以向老师获取 回归因为是一个值， 因此***\*模型最后输出是神经元节点数为1\****

***\*王道码农训练营\*******\*-WWW.CSKAOYAN.COM\****



 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps11.jpg)![img](file:///C:\Windows\Temp\ksohtml30060\wps12.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps13.jpg) 

过一个激活函数 relu

乘以一个系数

这里我们需要掌握如何将 ndarray 数据集变为 dataset，因为 dataset 可以直接 放入 DataLoader

帮助文档在 https://pytorch.org/docs/stable/data.html

 

 

Map-style datasets

 

A map-style dataset is one that implements

the __getitem__() and __len__() protocols, and represents a map from (possibly non-integral) indices/keys to data samples.

For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.

 

 

 

***\*DataLoader\**** ***\*的\**** ***\*shuffle\*******\*=\*******\*True\****

 

在 PyTorch 中 ，、DataLoader、是用于加载数据集的工具 ，其 中 、shuffle、参数决定了数据加载时是否对数据进行随机打乱。对于训练



集，、shuffle=True、和 、shuffle=False、会对训练过程产生不同的影响。

 

 

 

 

shuffle=True

 

\- **定义**: 在每个 epoch 之前，数据集都会被随机打乱。

 

\- **影响**:

 

\- **防止过拟合**: 数据的随机打乱可以防止模型记住数据的顺 序，从而降低过拟合的风险。

 

\- **更好地泛化能力**: 随机打乱数据有助于模型更好地泛化到未 见过的数据，因为它在训练时看到的数据顺序是不固定的。

 

\- **梯度下降效果更好**: 随机梯度下降（SGD）在处理随机数据 时通常效果更好，因为它可以更好地探索损失函数的不同区域，避免 陷入局部最小值。

 

 

 

 

shuffle=False

 

\- **定义**: 数据集在每个 epoch 之前不会被打乱，按照固定顺序加 载。

 

\- **影响**:

 

\- **固定数据顺序**: 模型在每个 epoch 看到的数据顺序是固定

 

 





 

的。

 

\- **潜在的过拟合风险**:  如果数据集有某种顺序，模型可能会记 住这种顺序，导致过拟合。

 

\- **梯度下降效果可能较差**:  在固定数据顺序下，梯度下降可能 会遇到一些问题，例如容易陷入局部最小值，训练过程可能不如随机 打乱时平稳。

 

 

 

 

对损失的影响

 

\- **shuffle=True**:  通常会导致损失函数的变化更加平滑，因为每个 batch  的数据是随机的，梯度下降会更稳定，损失函数会逐渐减少。

 

\- **shuffle=False**:  损失函数的变化可能会有一定的模式或周期性， 因为每个 epoch 的数据顺序是固定的。如果数据有某种顺序性，模 型可能会更快地在这种顺序上取得较低的训练损失，但这并不意味着 模型的泛化能力更好。

 

 

 

 

结论

 

\-  在训练集上，通常建议将 `shuffle=True` ，以便更好地训练模型并 提高其泛化能力。

 

\-  在验证集或测试集上，通常使用 `shuffle=False` ，以确保评估的一

 



 

致性。

 

 

 

***\*4\****  ***\*神经网络训练\*******\*-\*******\*理论\****

 

 

 

 

 

***\*梯度下降与反向传播\****

 

什么是前向传播，反向传播

![img](file:///C:\Windows\Temp\ksohtml30060\wps14.png) 

 

 

 

 

 

 

***\*1.\****  ***\*梯度是什么\*******\*?\****

 

梯度：是一个向量，导数 +  变化最快的方向(学习的前进方向)

 

回顾机器学习

 

收集数据x ，构建机器学习模型f ，得到f![img](file:///C:\Windows\Temp\ksohtml30060\wps15.jpg)x, w![img](file:///C:\Windows\Temp\ksohtml30060\wps16.jpg) = ypredict 判断模型好坏的方法：

 





 

| loss  = (ypredict — ytrue)**2** | (**回归损失**) |
| ------------------------------- | -------------- |
| loss   = ytrue  . log(ypredict) | (**分类损失**) |

 

 

 

 

目标：通过调整(学习)参数w ，尽可能的降低loss ，那么我们该如何 调整w呢？

![img](file:///C:\Windows\Temp\ksohtml30060\wps17.png) 

 

 

随机选择一个起始点w**0**,通过调整w**0** ，让 loss 函数取到最小值

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps18.png) 

 

 

w ***\*的更新方法\****：

 

1. 计算w 的梯度（导数）--计算机如何去算（近似求导）

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps19.jpg) 

 

1. 更新w

w = w — α ▽w

 

其中：

 

1. ▽w < **0** ,意味着 w 将增大
2. ▽w > **0** ,意味着 w 将减小

 

总结：梯度就是多元函数参数的变化趋势（参数学习的方向），只有 一个自变量时称为***\*导数\****

 

 

 

 

 

 

 



 

***\*2.\****  ***\*偏导的计算\****

 

***\*2.1  常见的导数计算\****

 

•  多项式求导数：f ![img](file:///C:\Windows\Temp\ksohtml30060\wps20.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps21.jpg) = x**5**  ,f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps22.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps23.jpg) = **5**x(**5**—**1**)

•  基本运算求导：f ![img](file:///C:\Windows\Temp\ksohtml30060\wps24.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps25.jpg) = xy ，f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps26.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps27.jpg) = y

•  指数求导：f ![img](file:///C:\Windows\Temp\ksohtml30060\wps28.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps29.jpg) = **5**ex  ，f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps30.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps31.jpg) = **5**ex

•  对数求导：f ![img](file:///C:\Windows\Temp\ksohtml30060\wps32.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps33.jpg) 表示 log 以 e 为底的对数

• 导数的微分形式：

f′ (x![img](file:///C:\Windows\Temp\ksohtml30060\wps34.jpg) =     ![img](file:///C:\Windows\Temp\ksohtml30060\wps35.jpg)

**牛顿**  **莱布尼兹**

 

那么：如何求f![img](file:///C:\Windows\Temp\ksohtml30060\wps36.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps37.jpg) = ![img](file:///C:\Windows\Temp\ksohtml30060\wps38.jpg)**1** + e—x)—**1**  的导数呢？那就可以使用

 

 

 

 

f ![img](file:///C:\Windows\Temp\ksohtml30060\wps39.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps40.jpg) = ![img](file:///C:\Windows\Temp\ksohtml30060\wps41.jpg)**1** + e—x)—**1**  ==>  f ![img](file:///C:\Windows\Temp\ksohtml30060\wps42.jpg)a![img](file:///C:\Windows\Temp\ksohtml30060\wps43.jpg) = a—**1** , a ![img](file:///C:\Windows\Temp\ksohtml30060\wps44.jpg)b![img](file:///C:\Windows\Temp\ksohtml30060\wps45.jpg) = ![img](file:///C:\Windows\Temp\ksohtml30060\wps46.jpg)**1** + b![img](file:///C:\Windows\Temp\ksohtml30060\wps47.jpg) , b ![img](file:///C:\Windows\Temp\ksohtml30060\wps48.jpg)c![img](file:///C:\Windows\Temp\ksohtml30060\wps49.jpg) = ec, c ![img](file:///C:\Windows\Temp\ksohtml30060\wps50.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps51.jpg) = — x

则有：链式求导

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps52.jpg) 

= — a—**2**  × **1** × ec  × ![img](file:///C:\Windows\Temp\ksohtml30060\wps53.jpg)—**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps54.jpg)

= — (**1** + e—x)—**2** × e—x × ![img](file:///C:\Windows\Temp\ksohtml30060\wps55.jpg)—**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps56.jpg)

= e—x (**1** + e—x)—**2**

 

 

 

 

***\*2.2  多元函数求偏导\****

 

一元函数，即有一个自变量。类似f![img](file:///C:\Windows\Temp\ksohtml30060\wps57.jpg)x![img](file:///C:\Windows\Temp\ksohtml30060\wps58.jpg)

 



 

多元函数，即有多个自变量。类似f(x, y, z) , **三****个自变量**x, y, z

 

多元函数求偏导过程中：***\*对某一个自变量求导，其他自变量当做常量\**** ***\*即可\****

例 1：

f(x, y, z)   =  ax + by + cz

![img](file:///C:\Windows\Temp\ksohtml30060\wps59.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps60.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps61.jpg) 

 

例 2：

f(x, y)   =  xy

![img](file:///C:\Windows\Temp\ksohtml30060\wps62.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps63.jpg) 

 

例 3：

 

f ![img](file:///C:\Windows\Temp\ksohtml30060\wps64.jpg)x, w![img](file:///C:\Windows\Temp\ksohtml30060\wps65.jpg)    =  (y — xw)**2**

![img](file:///C:\Windows\Temp\ksohtml30060\wps66.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps67.jpg) 

 

***\*练习：\****

 

已知J(a, b, c![img](file:///C:\Windows\Temp\ksohtml30060\wps68.jpg) = **3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps69.jpg)a + bc) , **令**u = a + v, v = bc,求 a，b，c 各自的偏导数。

 

 

 

 





**令**:  J(a, b, c) = **3**u

![img](file:///C:\Windows\Temp\ksohtml30060\wps70.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps71.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps72.jpg) 

 

 

 

 

 

 

 

 

***\*3.\****  ***\*反向传播算法\****

 

***\*3.1\**** ***\*计算图和反向传播\****

 

计算图：通过图的方式来描述函数的图形

 

在上面的练习中，J(a, b, c![img](file:///C:\Windows\Temp\ksohtml30060\wps73.jpg) = **3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps74.jpg)a + bc) , **令**u = a + v, v = bc,把它绘制 成计算图可以表示为：

![img](file:///C:\Windows\Temp\ksohtml30060\wps75.png) 

 

绘制成为计算图之后，可以清楚的看到向前计算的过程

之后，对每个节点求偏导可有：

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps77.png) 

 

 

那么反向传播的过程就是一个上图的从右往左的过程， 自变量a, b, c 各自的偏导就是连线上的梯度的乘积（***\*这就是链式法则，链式求导\****）：

![img](file:///C:\Windows\Temp\ksohtml30060\wps78.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps79.jpg) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps80.jpg) 

 

 

 

 

 

 

 

 

 

 

 

 

 

 





 

***\*3.1.1\****  ***\*存在激活函数怎么办\****

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps81.png) 

每次这么一步一步去算 sigmoid 函数的导数好麻烦啊，能不能整体去计算呢，当然可以

![img](file:///C:\Windows\Temp\ksohtml30060\wps82.png) 

 

 

 

 

 



 

***\*3.1.2\****  ***\*各种门单元的作用是什么\****

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps84.png) 

加法门单元 x+y=q ，那么q 对 x ，对 y 求偏导到是 1 ，因此影响分配是均等的 Max 门单元，因为选择较大的，梯度会直接传递给了 z ，w 直接失效的

x*y=q, 因为 q/x 求偏导是 y ，q/y 求偏导是 x ，所以相当于互换的操作

 

红色位置的值是最终结果对当前位置求的偏导数的值，只能解释 x 变化了，最终值发生 了什么变化，变大或者变小，不适合解释损失达到最小的问题，但是如果我们知道了变化方 向，就知道了调整 x 的方法，因为我们希望最终值接近某一个目标值，按梯度方向变化是最 佳。

 

***\*3.2\**** ***\*神经网络中的反向传播\****

 

 

 

 

 

***\*3.2.1\****  ***\*神经网络的示意图\****

 

 

w**1**, w**2**, . . . . wn表示网络第 n 层权重

 

wn [i, j]表示第 n 层第 i 个神经元，连接到第 n+1 层第j个神经元的权 重。

 

 



 

![img](file:///C:\Windows\Temp\ksohtml30060\wps86.png) 

 

 

***\*3.2.2\****  ***\*神经网络的计算图\****

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps87.png) 

 

 

其中：

 

1. ▽0ut是根据损失函数对预测值进行求导得到的结果
2. f 函数可以理解为激活函数

 

 

 

 

 

 

 

***\*问题：\****那么此时w**1** [**1**,**2**]的偏导该如何求解呢？





 

通过观察，发现从out 到w**1** [**1**,**2**]的来连接线有两条

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps88.png) 

 

 

结果如下：

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps89.jpg)![img](file:///C:\Windows\Temp\ksohtml30060\wps90.jpg) = x**1** * f′ (a**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps91.jpg)

\* ![img](file:///C:\Windows\Temp\ksohtml30060\wps92.jpg)W**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps93.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps94.jpg) * f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps95.jpg)b**1**) * W**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps96.jpg)**1**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps97.jpg) * ▽out + W**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps98.jpg)**2**,**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps99.jpg) * f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps100.jpg)b**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps101.jpg)

\* W**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps102.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps103.jpg) * ▽out) 公式分为两部分：

 

1. 括号外：左边红线部分
2. 括号内
3. 加号左边：右边红线部分
4. 加号右边：蓝线部分

 

但是这样做，当模型很大的时候，计算量非常大

 

***\*所以反向传播的思想就是对其中的某一个参数单独求梯度，之后更\****





 

***\*新\****，如下图所示：

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps104.png) 

 

 

 

 

 

 

计算过程如下

 

▽w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps105.jpg)**1**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps106.jpg) = f![img](file:///C:\Windows\Temp\ksohtml30060\wps107.jpg)b**1** ![img](file:///C:\Windows\Temp\ksohtml30060\wps108.jpg) * ▽out       （**计算**w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps109.jpg)**1**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps110.jpg) **梯度**）

▽w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps111.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps112.jpg) = f![img](file:///C:\Windows\Temp\ksohtml30060\wps113.jpg)b**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps114.jpg) * ▽out       （**计算**w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps115.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps116.jpg) **梯度**）

 

▽b**1** = f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps117.jpg)b**1** ![img](file:///C:\Windows\Temp\ksohtml30060\wps118.jpg) * w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps119.jpg)**1**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps120.jpg) * ▽out  （**计算**w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps121.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps122.jpg) **梯度**）

▽b**2** = f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps123.jpg)b**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps124.jpg) * w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps125.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps126.jpg) * ▽out  （**计算**w**3** ![img](file:///C:\Windows\Temp\ksohtml30060\wps127.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps128.jpg) **梯度**） 更新参数之后，继续反向传播

 

 

 

 

 

 

 

 

 

 

 

 



 

![img](file:///C:\Windows\Temp\ksohtml30060\wps129.png) 

 

 

 

计算过程如下：

 

▽W**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps130.jpg)**1**,**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps131.jpg) = f![img](file:///C:\Windows\Temp\ksohtml30060\wps132.jpg)a**1** ![img](file:///C:\Windows\Temp\ksohtml30060\wps133.jpg) * ▽b**2**

▽a**2** = f′ ![img](file:///C:\Windows\Temp\ksohtml30060\wps134.jpg)a**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps135.jpg) * ![img](file:///C:\Windows\Temp\ksohtml30060\wps136.jpg)w**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps137.jpg)**2**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps138.jpg) ▽b**1** + W**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps139.jpg)**2**,**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps140.jpg) ▽b**2** ![img](file:///C:\Windows\Temp\ksohtml30060\wps141.jpg) 继续反向传播

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 



 

![img](file:///C:\Windows\Temp\ksohtml30060\wps142.png) 

 

 

计算过程如下：

 

▽W**1** ![img](file:///C:\Windows\Temp\ksohtml30060\wps143.jpg)**1**,**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps144.jpg) = x**1** * ▽a**2**

▽x**1** = (W**1** ![img](file:///C:\Windows\Temp\ksohtml30060\wps145.jpg)**1**,**1**![img](file:///C:\Windows\Temp\ksohtml30060\wps146.jpg) * ▽a**1** + w**1** ![img](file:///C:\Windows\Temp\ksohtml30060\wps147.jpg)**1**,**2**![img](file:///C:\Windows\Temp\ksohtml30060\wps148.jpg) * ▽a**2**) * x**1** , ***\*通用的描述如下\****

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps149.png) 

https://blog.csdn.net/bitcarmanlee/article/details/78819025（有时间可以看下）

大家也许已经注意到，这样做是十分冗余的， 因为很多路径被重复访问了 。 比如上图中，

a-c-e 和 b-c-e就都走了路径 c-e。对于权值动则数万的深度模型中的神经网络，这样的冗余 所导致的计算量是相当大的。

 

***\*同样是利用链式法则，\**** ***\*BP\**** ***\*算法则机智地避开了这种冗余，\**** ***\*它对于每一个路径只访问一次就\**** ***\*能求顶点对所有下层节点的偏导值。\****

 

 





 

***\*4 pytorch\**** ***\*内部求梯度做的优化\****

 

 

PyTorch  使用自动微分（Automatic Differentiation）技术来计算梯度， 并且提供了一些内部的优化机制来加速和改进梯度计算的效率。以下 是 PyTorch 内部求梯度时所做的一些优化：

 

 

 

基于计算图的梯度计算：PyTorch  使用***\*计算图来记录模型的前向传播\**** ***\*过程，并在反向传播时根据链式法则计算梯度\****。通过构建计算图， PyTorch  可以在反向传播过程中自动计算梯度，无需手动推导和实现 梯度计算过程。

 

 

 

延迟执行（Deferred Execution）：PyTorch  延迟执行的特性使得梯度 计算可以按需执行，而不是在每个操作上立即计算梯度。这种延迟执 行的机制可以减少不必要的梯度计算，提高计算效率。

 

 

 

高效的梯度计算算法：PyTorch  使用高效的梯度计算算法来加速梯度 计算过程。例如，PyTorch  使用反向自动微分（Reverse Mode Automatic Differentiation）来计算梯度，该算法通过计算图的反向遍历来计算梯 度，避免了显式地计算所有中间变量的梯度。





 

梯度优化算法：PyTorch  提供了一些内置的梯度优化算法，如随机梯 度下降（SGD）、Adam 、Adagrad 等。这些优化算法在计算梯度的 基础上，根据梯度的方向和大小来更新模型的参数，以最小化损失函 数。

 

 

 

分布式梯度计算：PyTorch  支持分布式训练，可以将梯度计算分布到 多个设备或计算节点上进行并行计算。这种分布式的梯度计算可以加 速训练过程，并处理大规模的数据和模型。

 

 

 

这些优化机制使得 PyTorch  在梯度计算方面具有高效性和灵活性， 从而能够支持复杂的深度学习模型训练和优化。

 

 

 

***\*PyTorch\****  ***\*支\**** ***\*持\**** ***\*两\**** ***\*种\**** ***\*类\**** ***\*型\**** ***\*的\**** ***\*微\**** ***\*分\**** ***\*：\**** ***\*数\**** ***\*值\**** ***\*微\**** ***\*分\**** ***\*（\**** ***\*Numerical\**** ***\*Differentiation\*******\*）和符号微分（\*******\*Symbolic Differentiation\*******\*）。\****

 

 

 

数值微分（近似求导）：数值微分是使用近似方法计算导数的一种技 术。在数值微分中，通过在函数的某个点附近进行有限差分来估计导 数。PyTorch 提供了数值微分的功能，可以使用 torch.autograd.grad 函数来计算函数的数值导数。***\*数值微分的优点\*******\*是简单易用，适用于任\**** ***\*何函数，但是计算精度可能受到数值误差的影响\****。





 

 

 

 

符号微分（代数求导）：符号微分是使用代数方法计算导数的一种技 术。在符号微分中，通过对函数的符号表达式进行求导来得到导数的 符号表达式。***\*PyTorch\**** ***\*在计算图构建过程中使用符号微分来自动计\**** ***\*算梯度\****。通过在计算图中跟踪操作的导数计算，PyTorch  可以在反向 传播过程中自动计算梯度。符号微分的优点是精确性和效率，但对于 复杂的函数，求导的符号表达式可能会变得复杂。

 

 

 

PyTorch  的自动微分（Automatic Differentiation）是基于符号微分的， 它利用计算图构建的过程来跟踪操作的导数计算。在 PyTorch 中， 用户只需定义模型的前向传播过程，PyTorch  会自动构建计算图并在 反向传播时计算梯度。这种自动微分的机制使得 PyTorch 可以方便 地进行梯度计算，并且支持复杂的深度学习模型的训练和优化。

 

 

 

总结来说，数值微分是通过近似方法计算导数，适用于任何函数，但 精度可能受到数值误差的影响。符号微分是通过代数方法计算导数， 提供了精确性和效率，但对于复杂函数，求导的符号表达式可能会变 得复杂。PyTorch  的自动微分基于符号微分，通过计算图构建和反向 传播来实现梯度计算。

 



 

在某些情况下，PyTorch 可能使用数值微分来验证梯度计算的正确 性。这通常***\*发生在用户自定义的操作或函数上\****，当无法通过符号微分 计算导数时，PyTorch  可能会使用数值微分进行梯度的数值验证。然 而，这种情况是为了验证梯度计算的正确性，而不是作为主要的梯度 计算方法。

***\*激活函数\****

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps150.png) 

为什么需要激活函数， 改为 selu 后 ，可以缓解梯度消失

 

***\*selu\**** ***\*相对于\**** ***\*elu\**** ***\*乘以\**** ***\*lamda\****

![img](file:///C:\Windows\Temp\ksohtml30060\wps151.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps152.png) 

不同激活函数的优缺点

***\*Sigmoid\**** ***\*函数饱和使梯度消失。\****当神经元的激活在接近 0 或 1 处时会饱和，***\*在这些区域梯度\**** ***\*几乎为\**** ***\*0\**** ***\*，这就会导致梯度消失\****，几乎就有没有信号（信号来源于损失的变化）通过神经传





回上一层。

Sigmoid 函***\*数的输出不是零中心的\****。因为如果输入神经元的数据总是正数，那么关于 w 的梯 度在反向传播的过程中，将会要么全部是正数，要么全部是负数，这将会导致梯度下降权重 更新时出现 z 字型的下降。

 

 

Tanh 解决了 Sigmoid 的输出是不是零中心的问题，但仍然存在饱和问题。

为了防止饱和，现在主流的做法会在激活函数前多做一步 batch normalization ，尽可能保证 每一层网络的输入具有均值较小的、零中心的分布。

 

 

ReLU 非线性函数图像如下图所示。相较于 sigmoid 和 tanh 函数，ReLU 对于***\*随机梯度下降\**** ***\*的收敛有巨大的加速作用\****；sigmoid 和tanh 在求导时含有指数运算，而 ***\*ReLU\**** ***\*求导几乎不存\**** ***\*在任何计算量。\****

对比 sigmoid 类函数主要变化是：

1）单侧抑制；

2）相对宽阔的兴奋边界；

3）稀疏激活性。

存在问题：

ReLU 单元比较脆弱并且可能“死掉 ”，而且是不可逆的，因此导致了数据多样化的丢失。 通过合理设置学习率，会降低神经元“死掉 ”的概率。(死掉是指在为负时输出为零）

 

 

LeakRelu 其中 ε是很小的负数梯度值，比如 0.01，Leaky ReLU 非线性函数图像如下图所示。 这样做目的是使负轴信息不会全部丢失，***\*解决了\*******\*ReLU\**** ***\*神经元“死掉\**** ***\*”的问题\****。更进一步 的方法是 PReLU ，即把 ε 当做每个神经元中的一个参数，是可以通过梯度下降求解的。

ELU 函数的公式为：f(x) = {x, x>=0; a(exp(x)-1), x<0} ，其中 a 是一个正数，通常取 1 。ELU 函数在 x<0 时***\*具有负指数特性\****，可以避免 ReLU 函数的死亡神经元问题，并且对于一些复杂 的任务，效果会略微好于 ReLU 函数。（ELU 相对 LeakRelu 在负值计算时间更长）

 

 

SELU 函数则是在 ELU 函数的基础上提出的改进方法， 旨在通过对网络的初始化和激活函 数进行约束来解决深层神经网络中的***\*梯度消失和梯度爆炸\****等问题。具体地，SELU 函数要求 网络满足若干假设条件（例如权重应该服从一定的高斯分布），并且在这些条件成立的情况 下，保证网络的输出服从一定的分布。

 

SELU 函数则需要满足若干假设条件，因此并不是适用于所有类型的神经网络。对于那些“标 准 ”的神经网络结构（例如MLP 或CNN），使用 SELU 函数可能会带来较好的性能提升， 但对于其他类型的网络（例如 LSTM 或GAN），使用 SELU 函数的效果则可能不如预期。





 

 

(下面的链接大家自行阅读）

https://www.cnblogs.com/wqbin/p/11099612.html

https://finance.sina.com.cn/tech/2021-02-24/doc-ikftssap8455930.shtml

 

***\*归一化与批归一化（批标准化）\****

 

归一化.

◆Min-max 归一化: x*=(x- min)/(max-min)

◆Z-score 标准化: x*=(x- u)/ σ

***\*批归一化\****

◆每层的激活值都做归一化，归一化就是我们之前做过的，得到的效果的形象图如下图所示

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps153.png) 

为什么需要批归一化，能够解决哪些问题？从以下六个方面来阐述批归一化为什么有如此好 的效力：

（1）激活函数

（2）优化器

（3）批量大小

（4）数据分布不平衡

（5）BN 解决了梯度消失的问题

 

***\*为什么需要归一化？\****

 

通过使用 BN ，每个神经元的激活变得（或多或少）高斯分布，即它通常中等活 跃，有时有点活跃，罕见非常活跃。协变量偏移是不满足需要的，因为后面的层 必须保持适应分布类型的变化（而不仅仅是新的分布参数，例如高斯分布的新均 值和方差值）。

 

 



 

神经网络学习过程本质上就是为了学习数据分布，如果训练数据与测试 数据的分布不同，网络的泛化能力就会严重降低。

输入层的数据，已经归一化，后面网络每一层的输入数据的分布一直在发生变化， 前面层训练参数的更新将导致后面层输入数据分布的变化，必然会引起后面每一 层输入数据分布的改变。而且，网络前面几层微小的改变，后面几层就会逐步把 这种改变累积放大。训练过程中网络中间层数据分布的改变称之为："Internal

Covariate Shift(***\*内部协变量偏移\****)" 。BN 的提出，就是要解决在训练过程中，中 间层数据分布发生改变的情况。

 

减均值除方差：***\*远离饱和区\****

 

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps154.png) 

 

中间层神经元激活输入x 从变化不拘一格的正态分布通过 BN 操作拉回到了均值 为 0 ，方差为 1 的高斯分布。这有两个好处：***\*1\**** ***\*、避免分布数据偏移；\*******\*2\**** ***\*、远离导\**** ***\*数饱和区。\****

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps155.png) 

 



但这个处理对于在-1~1 之间的梯度变化不大的激活函数，效果不仅不好，反而 更差。比如 sigmoid 函数，s 函数在-1~1 之间几乎是线性，BN 变换后就没有达 到非线性变换的目的；而对于 relu ，效果会更差，因为会有一半的置零。总之换 言之，***\*减均值除方差操作后可能会削弱网络的性能。\****

 

 

***\*缩放加移位\****

 

因此，必须进行一些转换才能将分布从 0 移开。使用缩放因子γ和移位因子β来 执行此操作。下面就是加了缩放加移位后的 BN 完整算法。

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps157.png) 

 

 

***\*批归一化的参数不调整，因为无需对应公式中的\**** ***\*r\**** ***\*和β在接口中的参数\**



 

***\*1\**** ***\*激活函数\****

 

在所有情况下，BN 都能显著提高训练速度

 

如果没有 BN ，使用 Sigmoid 激活函数会有严重的梯度消失问题

 

如下图所示，激活函数 sigmoid、tanh、relu 在使用了 BN 后，准确度都有显著 的提高（虚线是没有用 BN 的情况，实线是对应的使用 BN 的情况）。

![img](file:///C:\Windows\Temp\ksohtml30060\wps159.png) 

 

***\*2\****  ***\*优化器（优化器会在后面详解）\****

 

Adam 是一个比较犀利的优化器，但是如果普通的优化器 , 比如随机梯度下降法， 加上 BN 后，其效果堪比 Adam。

 

ReLU +Adam≈ReLU+ SGD + BN

 

***\*所以说，使用\**** ***\*BN\**** ***\*，优化器的选择不会产生显着\*******\*差异\****

 

 



 

***\*3\****  ***\*批量大小\*******\*(\*******\*batch\*******\*_\*******\*size\*******\*)\****

 

对于小批量（即4），BN 会降低性能，所以要避免太小的批量，才能保证批归 一化的效果。

***\*4\****  ***\*数据不平衡\****

 

但是，如果对于具有分布极不平衡的二分类测试任务（例如，99：1 ），BN 破 坏性能并不奇怪。也就是说，这种情况下不要使用 BN。

 

 

***\*5\**** ***\*BN\**** ***\*解决了梯度消失的问题\****

 

BN 很好地解决了梯度消失问题，这是由前边说的减均值除方差保证的，把每一 层的输出均值和方差规范化，***\*将输出从饱和区拉倒了非饱和区\****（导数），很好的 解决了梯度消失问题

 

 

请看下面链接

https://baijiahao.baidu.com/s?id=1621528466443988599&wfr=spider&for=pc

 

***\*Dropout\****

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps160.png) 

 



 

 

 

 

 

Dropout 的作用

***\*◆防止过拟合\****

◆训练集上很好,测试集.上不好:

◆参数太多,记住训练样本,不能泛化

 

例如不让任意两个样本发生组合，从而避免过拟合，这是一种

https://blog.csdn.net/qq_34216467/article/details/83141837

 

AlphaDropout 是 Dropout 的一个变种，它在应用 Dropout 时使用了一个额外的缩放因子 alpha。

与标准 Dropout 相比，***\*AlphaDropout\**** ***\*在将一些输出设置为零的同时，还会对剩余的输出\**** ***\*进行缩放，\**** ***\*以保持网络权重的期望值不变\****。

这种缩放可以保持网络的动态范围，有助于训练过程的稳定性。

AlphaDropout 通常用于带有 LeakyReLU 或 Parametric ReLU 激活函数的网络，因为这些 激活函数在负输入时不饱和，Dropout 可能会导致这些神经元的权重更新不足。

 

***\*5\**** ***\*实战内容\****

 

***\*实现深度神经网络\****

***\*更改激活函数\****

***\*实现批归一化\****

***\*实现\**** ***\*dropout\****

 

04_classification_model-dnn.ipynb

 

通过 for 循环， 实现 20 层深度学习神经网络

 

for i in range(1, layers_num) :

self.linear_relu_stack.add_module(f"Linear_ {i}", nn.Linear(100, 100))

self.linear_relu_stack.add_module(f"relu", nn.ReLU())

 

 

我们主要看 val_loss。

一般 val_loss 小 loss 大的情况不出现。

val_loss 小， loss 小是理想情况。

 





 

val_loss 大， loss 小是***\*过拟合。\****

val_loss 大， loss 大是没收敛 ，***\*欠\*******\*拟合。\****

![img](file:///C:\Windows\Temp\ksohtml30060\wps162.png) 

 

***\*心跳问题\****

 

 

 

 

训练时损失出现心跳（loss  oscillations）通常是由于学习率设置过高或批次 大小设置不合适导致的。学习率过高将导致模型参数在训练过程中发生剧烈波 动，从而导致损失值出现心跳；批次大小设置不合适可能会导致噪声数据对训练 过程产生影响，也会导致损失值出现心跳。

解决方法包括：

 

降低学习率：降低学习率可以减少模型参数的剧烈波动，有助于缓解损失值的心 跳问题。

调整批次大小：增加批次大小可以减少噪声数据对训练的影响，有助于减轻损失 值的心跳问题。

使用正则化技术：正则化技术可以帮助控制模型参数的范围，减少其波动性，从 而减轻损失值的心跳问题。

增加训练数据量：增加训练数据量可以降低模型的过拟合程度，有助于稳定损失 值的变化。

对模型架构进行调整：有时候，损失值心跳的问题可能是由于模型过于复杂或架 构设计不佳导致的。因此，对模型架构进行调整可能有助于缓解损失值心跳的问 题。

总之，解决训练时损失值心跳的问题需要综合考虑多个因素，并根据实际情况进 行适当的调整和优化。





 

***\*批归一化\****

 

接口

 

nn.BatchNorm1d

 

规范化和缩放输入或激活

实例

05_classification_model-dnn-bn .ipynb 官网  ---查看批归一化的公式与参数

![img](file:///C:\Windows\Temp\ksohtml30060\wps163.png) 

 

博客

https://blog.csdn.net/tefuirnever/article/details/93457816

 

***\*批归一化可以解决梯度爆炸问题\****

https://zhuanlan.zhihu.com/p/261866923

 

***\*梯度消失和梯度爆炸\****

 

什么是梯度消失和梯度爆炸？

***\*层数比较多的神经网络模型\****在训练时也是会出现一些问题的，其中就包括梯度消失问题 （gradient vanishing problem）和梯度爆炸问题（gradient exploding problem） 。梯度消失问 题和梯度爆炸问题一般随着网络层数的增加会变得越来越明显。

例如，对于下图所示的含有 3 个隐藏层的神经网络，梯度消失问题发生时，接近于输出层的 hidden layer 3 等的权值更新相对正常，但前面的 hidden layer 1 的权值更新会变得很慢，导 致前面的层权值几乎不变，仍接近于初始化的权值，这就导致 hidden layer 1 相当于只是一 个映射层，对所有的输入做了一个同一映射，这是此深层网络的学习就等价于只有后几层的 浅层网络的学习了。

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps164.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps165.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps166.png) 

 

 

 

 

 

 

 

 

 





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps167.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps168.png) 

![img](file:///C:\Windows\Temp\ksohtml30060\wps169.png) 

其实梯度爆炸和梯度消失问题都是因为网络太深，网络权值更新不稳定造成的，***\*本质上是因\**** ***\*为梯度反向传播中的连乘效应\****。对于更普遍的梯度消失问题，可以考虑用ReLU 激活函数取 代 sigmoid 激活函数。

 

梯度消失

https://zhuanlan.zhihu.com/p/25631496   梯度消失和梯度爆炸是什么

https://www.jianshu.com/p/3f35e555d5ba  通过哪些方法去解决梯度消失和爆炸

https://www.jianshu.com/p/cb1fa15d32d8   批归一化公式 梯度爆炸

https://zhuanlan.zhihu.com/p/32154263  解决方法

 

 

***\*梯度消失\****

 

梯度消失问题发生时，接近于输出层的 hidden layer 3 等的权值更新相对正常，但前面的 hidden layer 1 的权值更新会变得很慢，导致前面的层权值几乎不变，仍接近于初始化的权值，





这就导致 hidden layer 1 相当于只是一个映射层，对所有的输入做了一个同一映射，这是此 深层网络的学习就等价于只有后几层的浅层网络的学习了

 

如何解决：

***\*1\****  ***\*换用\**** ***\*Relu\**** ***\*、\*******\*LeakyRelu\**** ***\*、\*******\*Elu\**** ***\*等激活函数\****

ReLu：让激活函数的导数为 1

LeakyReLu：包含了 ReLu 的几乎所有有点，同时解决了 ReLu 中 0 区间带来的影响

ELU：和 LeakyReLu 一样，都是为了解决 0 区间问题，相对于来，elu 计算更耗时一些（为 什么）

***\*2 BatchNormalization\****

BN 本质上是解决传播过程中的梯度问题

***\*3\**** ***\*ResNet\**** ***\*残差结构（后面讲解）\****

***\*4\**** ***\*LSTM\**** ***\*结构（后面讲解）\****

LSTM 不太容易发生梯度消失，主要原因在于 LSTM 内部复杂的“门（gates）”

***\*5\****  ***\*预训练加\**** ***\*finetunning\*******\*（精调\*******\*）（\*******\*后面实战）\****

Hinton 在 06 年发表的论文上，其基本思想是每次训练一层隐藏层节点，将上一层隐藏层的 输出作为输入，而本层的输出作为下一层的输入，这就是逐层预训练。

训练完成后，再对整个网络进行“微调（fine-tunning）” 。 此方法相当于是找全局最优，然后整合起来寻找全局最优

 

 

 

 

 

 

 

 

 

 

***\*梯度爆炸\****

 

在反向传播过程中使用的是***\*链式求导法则\****，如果每一层偏导数都大于 1 ，那么连乘起来将以 指数形式增加，误差梯度不断累积，就会造成梯度爆炸。梯度爆炸会导致模型权重更新幅度 过大，会造成模型不稳定，无法有效学习，还会出现无法再更新的 NaN 权重值。

 

模型无法在训练数据上收敛（比如，损失函数值非常差）；

模型不稳定，在更新的时候损失有较大的变化；

模型的损失函数值在训练过程中变成 NaN 值；

 

解决方法：

***\*1.\*******\*重新设计网络模型\****

在深层神经网络中，梯度爆炸问题可以通过将网络模型的***\*层数变少\****来解决。此外，在训练网 络时，使用较小批量（batch_size 不要太大）也有一些好处。

***\*2.\*******\*使用修正线性激活函数\****

在深度多层感知机中，当激活函数选择为一些之前常用的 Sigmoid 或 Tanh 时，网络模型会

***\*王道码农训练营\*******\*-WWW.CSKAOYAN.COM\****



发生梯度爆炸问题。而使用修正线性激活函数（ReLU）能够减少梯度爆炸发生的概率，对 于隐藏层而言，使用修正线性激活函数（ReLU）是一个比较合适的激活函数，当然 ReLU 函数有许多变体，大家在实践过程中可以逐一使用以找到最合适的激活函数。

***\*3.\*******\*使用长短周期记忆网络（\*******\*LSTM\*******\*）\*******\*---\*******\*后面讲\****

由于循环神经网络中存在的固有不稳定性，梯度爆炸可能会发生。比如，通过时间反向传播， 其本质是将循环网络转变为深度多层感知神经网络。通过使用长短期记忆单元（LSTM）或 相关的门控神经结构能够减少梯度爆炸发生的概率。（因为 LSTM 有梯度裁剪，检查误差 梯度值就是与一个阈值进行比较，若误差梯度值超过设定的阈值，则截断或设置为阈值。） 对于循环神经网络的时间序列预测而言，采用LSTM 是新的最佳实践。

***\*4.\*******\*使用权重正则化\*******\*---\*******\*后面实战\****

如果梯度爆炸问题仍然发生，另外一个方法是对网络权重的大小进行校验，并对大权重的损 失函数增添一项惩罚项，这也被称作权重正则化，常用的有 L1（权重的绝对值和）正则化 与 L2（权重的绝对值平方和再开方）正则化。（Dense 层中有参数设置正则化）

 

未进行批归一化的图

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps170.png) 

loss: 0.4376 accuracy: 0.8839

 

***\*SELU\**** ***\*实例\****

 

06_classification_model-dnn-selu.ipynb 我们也可以使用 selu 激活函数

比例指数线性单位

 

也能一定程度上缓解梯度消失的问题，***\*从训练时间上观察比批归\*******\*一化要快一些\**** ***\*Selu\**** ***\*可以用来解决梯度爆炸问题\****

 

博客

https://www.jianshu.com/p/d216645251ce

 

 





 

softmax 是多分类 ，sigmoid 是二分类

 

 

***\*SELU-DROPOUT 实例\****

 

 

07_classification_model-dnn-selu-dropout.ipynb 接口

nn.AlphaDropout

 

Alpha Dropout 是一种 Dropout 将输入的均值和方差保持为其原始值的方法 ， 即使在此 Dropout 之后也可以确保自规范化属性。通过将激活随机设置为负饱和值，Alpha Dropout 非常适合 比例指数线性单位。

 

Alpha dropout 相对于不同的 dropout ，在操作后，***\*激活函数的分布不会发生变化\****

 

为了防止过拟合，我们增加 dropout ，只在最后几层添加 dropout（也可以放在前几层）

 

我们设置为 0.5 ，因为当是 0.5 时它的（子网络的数目）是最大的。在用了 dropout 之后，因 为每次都是随机的丢弃一部分值，丢弃之后相当于一个子网络。而在丢弃率是 0.5 的时候， 可能的子网络的数目最大，这是因为排列组合，比如，一共有 4 个单元，我如果每次丢弃两 个，那么可能的选择是 C(2, 4)= 6 ， 而如果丢弃一个或三个，那么可能的选择就是 C(1, 4)= C(3, 4) = 4。

 

dropout 是应用在特征的激活值上，alpha-dropout 也是这样。dropout 是将要丢弃的激活值设 为 0, alpha-dropout 是将要丢弃的激活值设为 alpha。

具体细节可以参考论文：https://arxiv.org/pdf/1706.02515.pdf

 

因为普通的 dropout 是将被丢弃的数据点设为 0,  而 alpha dropout 则是设为 alpha，alpha 是变 化的，是按照分布不变的约束求出来的，所以分布保持不变。

 

在自然界中，在中大型动物中，一般是有性繁殖，有性繁殖是指后代的基因从父母两方各继 承一半。但是从直观上看，似乎无性繁殖更加合理，因为无性繁殖可以保留大段大段的优秀 基因。而有性繁殖则将基因随机拆了又拆，破坏了大段基因的联合适应性。

但是自然选择中毕竟没有选择无性繁殖，而选择了有性繁殖，须知物竞天择，适者生存。我 们先做一个假设，那就是基因的力量在于混合的能力而非单个基因的能力。不管是有性繁殖 还是无性繁殖都得遵循这个假设。为了证明有性繁殖的强大，我们先看一个概率学小知识。 比如要搞一次恐怖袭击，两种方式：

\-  集中 50 人，让这 50 个人密切精准分工，搞一次大爆破。

\-  将 50 人分成 10 组，每组 5 人，分头行事，去随便什么地方搞点动作，成功一次就算。 哪一个成功的概率比较大？ 显然是后者。因为将一个大团队作战变成了游击战。

那么，类比过来，有性繁殖的方式不仅仅可以将优秀的基因传下来，还可以降低基因之间的





联合适应性，使得复杂的大段大段基因联合适应性变成比较小的一个一个小段基因的联合适 应性。

dropout 也能达到同样的效果，它强迫一个神经单元，和随机挑选出来的其他神经单元共同 工作，达到好的效果。消除减弱了神经元节点间的联合适应性，***\*增强了泛化能力\****。

个人补充一点：那就是植物和微生物大多采用无性繁殖，因为他们的生存环境的变化很小， 因而不需要太强的适应新环境的能力，所以保留大段大段优秀的基因适应当前环境就足够 了。而高等动物却不一样，要准备随时适应新的环境，因而将基因之间的联合适应性变成一 个一个小的，更能提高生存的概率。

 

 

 

 

 

 

***\*我们使用了\**** ***\*dropout\**** ***\*以后没有原来的效果好，说明我们的数据的过拟合情\*******\*况比较轻\****

 

 

 

 

***\*6\**** ***\*Wide\**** ***\*&\**** ***\*Deep\**** ***\*模型\*******\*—\*******\*推荐场景\****

 

◆16 年发布，用于分类和回归

◆应用到了 Google Play 中的应用推荐

◆原始论文: https://arxiv.org/pdf/1606.07792v1.pdf

 

稀疏特征

◆离散值特征

◆One-hot 表示

◆Eg:  专业={计算机,人文，其他}.人文=[0, 1, 0]

◆Eg:词表={人工智能,你，他，王道,...他=[0,0, 1,0,..

***\*◆叉乘\*******\*={\*******\*计算机\*******\*,\*******\*人工智能\*******\*)\**** ***\*，\*******\*(\*******\*计算机\*******\*,\*******\*你\*******\*), ..\****

 

炸鸡与汉堡

https://blog.csdn.net/Yasin0/article/details/100736420

谷歌 app 应用商店

https://zhuanlan.zhihu.com/p/57247478

wide—deep 介绍（最佳）

https://zhuanlan.zhihu.com/p/139358172

 

***\*稀疏特征(Wide)\****

 

◆叉乘之后

稀疏特征做叉乘获取共现信息

◆实现记忆的效果





 

 

特征 1——专业: {计算机、人文、其他} ，特征 2——下载过音乐《消愁》:{是、否} ，这两 个特征 one-hot 后的特征维度分别为 3 维与2 维，对应的叉乘结果是特征 3——专业×下载 过音乐《消愁》: {计算机∧是，计算机∧否，人文∧是，人文∧否，其他∧是，其他∧否}。

 

我们对于任何组合都进行了训练，是不是就可以保存一个物体的所有可能性

 

 

优缺点

◆优点

有效，广泛用于工业界

主要用于 广告（谷歌，百度），亚马逊，淘宝，京东的业务

 

◆缺点

◆需要***\*人工设计\****

◆可能过拟合，所有特征都叉乘,相当于记住每一个样本（n 的 k 次方）

◆***\*泛化能力差\****,没出现过就不会起效果（我很开心，我很快乐是两个样本）

 

 

 

***\*密集特征（\**** ***\*Deep）\****

 

◆向量表达

◆Eg:  词表={人工智能,你,他,王道论坛}.

◆他=[0.3, 0.2, 0.6, (n 维向量)]

对于高维稀疏的分类特征，首先会转化成低维的稠密的向量

◆Word2vec 工具

◆男-女=国王-王后  通过距离来等价

我们就可以向量之间的差距来衡量信息之间的差距

 

***\*密集特征的优缺点\****

 

◆优点

◆带有语义信息,不同向量之间有相关性

◆兼容没有出现过的特征组合

◆更少人工参与.

◆缺点

◆过度泛化,推荐不怎么相关的产品

 

 

 

 

 

 





 

![img](file:///C:\Windows\Temp\ksohtml30060\wps171.png) 

多了特征的密集表达和中间层

![img](file:///C:\Windows\Temp\ksohtml30060\wps172.png) 

Embedding 是一个将离散变量转为连续向量表示的一个方式

 

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps173.png) 

 

 

 

 



***\*Wide &\**** ***\*Deep 模型实战\****

 

 

 

◆子类 API

◆功能 API (函数式 API )

◆多输入与多输出

 

08-09_regression-wide_deep.ipynb 函数式 API 搭建

 

我们往往是***\*面向对象编程\****，因此接下来我们通过子类搭建来完成wide-deep 模型

 

子类 API 搭建

实现是根据官网的帮助，按要求实现即可

 

10_regression-wide_deep-multi-input.ipynb 多输入

 

如果 wide&deep 模型的 wide 的 input 和 deep 的 input 不一样

 

 

11_regression-wide_deep-multi-output.ipynb 多输出

多输出并不是wide deep 特有的要求，而是我们深度学习可能有这样的场景，比如我们同时 预测当前的房价，以及一年后的房价，需要两个输出，因为我们目前没有第二年房价的数据， 因此暂时使用了同一个数据集，模拟一下

 

这里有一个练习wide-deep 的数据集，作为作业

https://blog.csdn.net/weixin_34268753/article/details/92123569

 

***\*7\****  ***\*超参数搜索\****

 

为什么要超参数搜索

◆超参数就是神经网络训练过程中不变的参数（变的参数，例如我们通过 fit 时，里边 的参数是变的，例如 w 和 b），***\*我们需要调的参数，就是超参数\****

***\*下面的这些才是超参数\****

◆网络结构参数:***\*几层（用什么样的层）\****,***\*每层神经元个数\****，***\*每层激活函\*******\*数\****等

◆训练参数: ***\*batch\*******\*_\*******\*size\**** ***\*,\*******\*学习率\****,***\*学习率衰减算法\****等

***\*◆手工去试耗费人力\****



 

 

Batch Size 定义 ：一次训练所选取的样本数。

下面链接详细讲解了 Batch Size

https://blog.csdn.net/qq_34886403/article/details/82558399

***\*Batch\**** ***\*Size\**** ***\*从小到大的变化对网络影响\****

***\*1\**** ***\*、没有\**** ***\*Batch\**** ***\*Size\**** ***\*，梯度准确，只适用于小样本数据库\****

***\*2\**** ***\*、\*******\*Batch\**** ***\*Size\*******\*=1\**** ***\*，梯度变来变去，非常不准确，网络很\*******\*难收敛（不存在）。\****

***\*3\**** ***\*、\*******\*Batch\**** ***\*Size\**** ***\*增大，梯度变准确，\****

***\*4\**** ***\*、\*******\*Batch\**** ***\*Size\**** ***\*增大，梯度已经非常准确，再增加\**** ***\*Batch\**** ***\*Size\**** ***\*也没有用\****

 

***\*注意：\*******\*Batch\**** ***\*Size\**** ***\*增大了，要到达相同的准确度，必须要增大\**** ***\*epoch\*******\*。\****

 

 

***\*学习率(\*******\*Learning\**** ***\*rate\*******\*)\****作为监督学习以及深度学习中重要的超参，其决定着目标函数能否 收敛到局部最小值以及何时收敛到最小值。***\*合适的\*******\*学习率能够使\**** ***\*目标函数\*******\*(\*******\*损失函数\*******\*)\****   ***\*在\**** ***\*合适的时间内收敛到局部最小值。\****

https://www.cnblogs.com/lliuye/p/9471231.html（这里有形象的图） 学习率衰减算法(后面会详细讲解)

![img](file:///C:\Windows\Temp\ksohtml30060\wps174.png) 

 

 

加快学习的一个办法就是随时间慢慢减少学习率 ，我们称之为学习率衰减

https://blog.csdn.net/u010132497/article/details/79746061

 

 

 

***\*搜索策略\****

 

◆网格搜索 （全组合）

◆随机搜索

◆遗传算法搜索





◆启发式搜索

 

 

***\*网格搜索\****

 

◆定义 n 维方格

◆每个方格对应- -组超参数

◆一组一组参数尝试

 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps175.png) 

 

Dropout rate

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

***\*随机搜索\****

 

参数的生成方式为随机 可探索的空间更大

https://www.jianshu.com/p/2cf662ada97e

![img](file:///C:\Windows\Temp\ksohtml30060\wps176.png) 

 

![img](file:///C:\Windows\Temp\ksohtml30060\wps177.png)***\*王道码农训练营\*******\*-WWW.CSKAOYAN.COM\****



 

***\*遗传算法\****

 

◆对自然界的模拟

◆A.初始化候选参数集合->训练->得到模型指标作为生存概率

◆B.选择->交叉->变异->产生下一代集合

◆C.重新到 A

 

 

下面根据上图来介绍遗传算法的基本概念，首先

 

1 、初始化种群，我们需要知道，种群包含的个体称为染色体，这个染色体上的每一个片段 都被称为一个基因，一条染色体对应了目标问题的一个可行解，以超参数优化为例，一条染 色体就是一组超参数，而超参数组合中的每一个超参数是一个基因；

 

2 、适应度函数，用于衡量每一条染色体的优劣程度，对应于超参数优化问题，适应度函数 实际上就是我们交叉验证的评价指标的值，这里根据用户所面临的问题由用户自己来定义；

 

3 、选择，优胜劣汰，适者生存，很充分的体现了选择的核心，对应到超参数优化种就是我 们每次迭代都会删除一部分表现最差的超参数组合，留下表现最好的部分超参数组合；

 

4 、交叉，对应了自然界中的交配的概念，品质优良的染色体之间互相交换基因组，就像是 孩子的基因是由父母决定的一样，对应到超参数优化中就是我们对选择之后剩下的最优秀的 超参数组合进行多次选择，选择出的两个超参数组合进行交叉形成新的超参数组合进入下一 个环节，需要注意的是，“父母 ”超参数组合并不是使用均匀的随机采样，而是基于轮盘赌 的算法。

 

 

 

下面有代码示例（有时间同学可以练习） https://zhuanlan.zhihu.com/p/123319468

 

***\*启发式搜索\*******\*.\****

 

◆研究热点-AutoML 自动调参与部署（缺点：耗费硬件成本）

◆使用循环神经网络来生成参数

***\*◆使用强化学习来进行反馈\*******\*,\*******\*使用模型来训练生成参数（\*******\*ChatGPT\**** ***\*用了，后面解析）\**** 基于环境，规则的随机尝试，奖励来判断是否最优

 

 

 

 

 

 

 

 





***\*参数搜索实战\****

 

12-13_regression-hp-search.ipynb

我们可以发现训练的学习率较小，训练速度较慢，学习率合适，就会达到一个好的效果，学 习率太大，就爆炸了

 

实际我们远不止一个参数，我们需要很多层 for 循环来测试，我们其实可以考虑并行化处理

 

 

 

 

 

***\*12-13_\*******\*regression\*******\*-\*******\*hp\*******\*-\*******\*search\*******\*.\*******\*ipynb\****

 

 

 

 

 

 

 

 

***\*关于代码多次执行后，结果存在不同的原因\**** 

https://cloud.tencent.com/developer/article/1065617

