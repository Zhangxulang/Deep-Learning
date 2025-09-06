# 一、Fashion-MNIST 简介

**Fashion-MNIST** 是由 Zalando 公司在 2017 年发布的一个图像分类数据集，主要用来替代过于简单的 **MNIST 手写数字数据集**。

- **目标**：提供一个更接近真实世界、但仍然简单易用的基准数据集。
- **类型**：灰度图像，二维矩阵。
- **任务**：图像分类（10 个类别的时尚物品）。

因为 MNIST 经过多年研究已经被“刷爆”（很多模型轻松 > 99% 准确率），研究人员需要一个更有挑战性，但格式兼容 MNIST 的数据集，所以有了 Fashion-MNIST。

------

# 二、数据集组成

Fashion-MNIST 与 MNIST 格式完全相同，因此替换无缝。

- **图片数量**：

  - 训练集：60,000 张
  - 测试集：10,000 张

- **图像大小**：28 × 28 像素

- **颜色通道**：灰度（单通道）

- **类别数**：10 类

- **类别标签**：

  | Label | 类别 (英文) | 类别 (中文) |
  | ----- | ----------- | ----------- |
  | 0     | T-shirt/top | T恤 / 上衣  |
  | 1     | Trouser     | 裤子        |
  | 2     | Pullover    | 套衫        |
  | 3     | Dress       | 连衣裙      |
  | 4     | Coat        | 外套        |
  | 5     | Sandal      | 凉鞋        |
  | 6     | Shirt       | 衬衫        |
  | 7     | Sneaker     | 运动鞋      |
  | 8     | Bag         | 包包        |
  | 9     | Ankle boot  | 短靴        |

- **文件格式**：与 MNIST 相同，包含四个文件（gz 压缩）

  - 训练集图像：`train-images-idx3-ubyte.gz`
  - 训练集标签`train-labels-idx1-ubyte.gz`
  - 测试集图像：`t10k-images-idx3-ubyte.gz`
  - 测试集标签`t10k-labels-idx1-ubyte.gz`

------

# 三、特点

1. **难度比 MNIST 大**
   - MNIST：很多模型轻松达到 >99%
   - Fashion-MNIST：同样模型一般只有 ~90% 左右
2. **格式兼容**：直接替换 MNIST，无需改代码。
3. **轻量级**：文件大小 ~30MB，易于下载和使用。
4. **应用场景**：常用于快速验证新算法、教学示例、深度学习入门。

------

# 四、常见使用方式

## 1. 使用 PyTorch

```python
import torch
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),               # 转换为张量，范围 [0,1]
    transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1,1]
])

# 加载训练集
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试集
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

## 2. 使用 TensorFlow / Keras

```python
import tensorflow as tf

# 直接加载
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

print("训练集:", train_images.shape, train_labels.shape)
print("测试集:", test_images.shape, test_labels.shape)
```

------

# 五、可视化示例

```py
import matplotlib.pyplot as plt

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap="gray")
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

------

# 六、应用场景

1. **深度学习入门实验**：替代 MNIST，增加挑战性。
2. **模型对比基准**：卷积神经网络 (CNN)、迁移学习等。
3. **教学与课程作业**：数据小、下载快、可视化直观。
4. **快速验证算法原型**：测试优化器、正则化、数据增强等。

------

# 七、总结

- **Fashion-MNIST** 是 MNIST 的升级版，难度更高，但格式相同。
- **数据特点**：28×28 灰度，10 类时尚物品，7 万张图。
- **用途**：深度学习入门、算法验证、教学演示。
- **优势**：小巧、开源、替换无缝、视觉识别难度更贴近真实场景。

