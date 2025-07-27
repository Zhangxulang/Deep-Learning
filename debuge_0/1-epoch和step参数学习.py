# 作者: 大数据 浪哥
# 2025年07月28日03时00分00秒
# 1054074422@qq.com

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 伪造训练数据：1000个样本，每个样本10个特征
x = torch.randn(1000, 10)  # 输入特征矩阵（1000行，10列）
y = torch.randn(1000, 1)   # 目标输出向量（1000行，1列）

# 用TensorDataset包装数据（方便DataLoader加载）
dataset = TensorDataset(x, y)

# DataLoader按batch_size=100分批读取数据，shuffle=True表示每个epoch打乱数据
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# 定义一个线性模型：输入10维，输出1维，相当于 y = Wx + b
model = nn.Linear(10, 1)

# 均方误差损失函数（MSE）：用于回归任务
loss_fn = nn.MSELoss()

# 优化器：使用随机梯度下降（SGD），学习率为0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 设置训练总轮数（epoch数量）
num_epochs = 5

# 进入训练循环，每次epoch表示完整训练集被模型“看”一遍
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")  # 打印当前是第几轮训练

    # 用 enumerate 给DataLoader加上 step 计数器（step从0开始）
    for step, (batch_x, batch_y) in enumerate(dataloader):
        # 前向传播：模型对当前batch进行预测
        preds = model(batch_x)

        # 计算预测值和真实值之间的损失
        loss = loss_fn(preds, batch_y)

        # 反向传播之前清空梯度
        optimizer.zero_grad()

        # 反向传播：计算梯度
        loss.backward()

        # 参数更新：根据梯度调整模型参数
        optimizer.step()

        # 打印当前step的信息，包括损失值
        print(f"  Step {step+1}, Loss: {loss.item():.4f}")
