# Deep-Learning

本项目记录了从零开始学习深度学习的全过程，包含理论知识、实战代码、环境配置及工具使用等内容，适合入门者系统学习和进阶者参考。

## 项目结构

### 1. 基础框架与核心概念

- **PyTorch 实战**：涵盖分类、回归任务，Wide&Deep 模型实现，超参数搜索（如 Batch Size、学习率调优），自定义损失函数与网络层等。
- **TensorFlow 与深度学习基础**：DNN 模型构建、梯度消失 / 爆炸解决方案、批归一化、Dropout 等核心技术。
- **张量操作与数据处理**：`torch.Tensor`操作、数据集加载（`Dataset`、`DataLoader`）、自定义数据集实现。

### 2. 经典模型与应用

- **RNN 与序列建模**：文本处理（词典构建、词嵌入）、情感分类、文本生成，LSTM 长短期记忆网络原理与实战。
- **Transformer 模型**：基于《Attention is All You Need》的手搓实现，包含多头注意力、层归一化、位置编码等核心模块，以及机器翻译实战（如德语 - 英语翻译）。
- **Wide&Deep 模型**：适用于推荐系统场景，结合稀疏特征叉乘（记忆能力）与深层网络（泛化能力），附实战案例与数据集。

### 3. 大语言模型与微调

- **LoRA 微调技术**：基于 LLaMA 模型的低秩适配微调详解，包括 QLoRA（4-bit 量化训练）、PEFT 工具库使用，以及部署注意事项（权重合并、性能测试）。
- **GPT 相关技术**：Prompt 工程（零样本 / 少样本学习）、数据集清洗（LSH 去重）、强化学习对齐（RLHF）等。

### 4. 云服务与环境配置

- 云平台使用
  - Kaggle/Colab：免费 GPU 资源、数据集下载与 API 使用。
  - 国内平台：百度 AI Studio、阿里云 DSW（适合国内网络环境）。
- 环境搭建
  - CUDA 与 GPU 加速：安装教程、版本匹配、cuDNN 配置。
  - PyTorch/TensorFlow 安装：依赖管理、虚拟环境配置。

### 5. 工具与资源

- **Hugging Face 生态**：`transformers`库、`tokenizers`（高性能分词工具）、模型权重与源码托管。
- **可视化与调试**：TensorBoard 使用、训练过程损失波动分析。

## 使用说明

1. 克隆仓库：

   ```bash
   git clone https://github.com/Zhangxulang/Deep-Learning.git
   cd Deep-Learning
   ```

2. 环境配置：参考`环境搭建.md`配置 CUDA、PyTorch 等依赖。

3. 实战练习：各目录下的`*.ipynb`文件可直接运行，部分脚本需参考对应`md`文档的步骤说明。

## 参考资料

- 论文：《Attention is All You Need》、《LoRA: Low-Rank Adaptation of Large Language Models》等。
- 工具文档：[PyTorch 官方文档](https://pytorch.org/docs/)、[Hugging Face PEFT](https://huggingface.co/docs/peft)。
- 数据集与平台：Kaggle 数据集、Hugging Face Model Hub。

欢迎通过 Issues 交流学习问题，或提交 PR 补充内容！# Deep-Learning

本项目记录了从零开始学习深度学习的全过程，包含理论知识、实战代码、环境配置及工具使用等内容，适合入门者系统学习和进阶者参考。

## 项目结构

### 1. 基础框架与核心概念

- **PyTorch 实战**：涵盖分类、回归任务，Wide&Deep 模型实现，超参数搜索（如 Batch Size、学习率调优），自定义损失函数与网络层等。
- **TensorFlow 与深度学习基础**：DNN 模型构建、梯度消失 / 爆炸解决方案、批归一化、Dropout 等核心技术。
- **张量操作与数据处理**：`torch.Tensor`操作、数据集加载（`Dataset`、`DataLoader`）、自定义数据集实现。

### 2. 经典模型与应用

- **RNN 与序列建模**：文本处理（词典构建、词嵌入）、情感分类、文本生成，LSTM 长短期记忆网络原理与实战。
- **Transformer 模型**：基于《Attention is All You Need》的手搓实现，包含多头注意力、层归一化、位置编码等核心模块，以及机器翻译实战（如德语 - 英语翻译）。
- **Wide&Deep 模型**：适用于推荐系统场景，结合稀疏特征叉乘（记忆能力）与深层网络（泛化能力），附实战案例与数据集。

### 3. 大语言模型与微调

- **LoRA 微调技术**：基于 LLaMA 模型的低秩适配微调详解，包括 QLoRA（4-bit 量化训练）、PEFT 工具库使用，以及部署注意事项（权重合并、性能测试）。
- **GPT 相关技术**：Prompt 工程（零样本 / 少样本学习）、数据集清洗（LSH 去重）、强化学习对齐（RLHF）等。

### 4. 云服务与环境配置

- 云平台使用
  - Kaggle/Colab：免费 GPU 资源、数据集下载与 API 使用。
  - 国内平台：百度 AI Studio、阿里云 DSW（适合国内网络环境）。
- 环境搭建
  - CUDA 与 GPU 加速：安装教程、版本匹配、cuDNN 配置。
  - PyTorch/TensorFlow 安装：依赖管理、虚拟环境配置。

### 5. 工具与资源

- **Hugging Face 生态**：`transformers`库、`tokenizers`（高性能分词工具）、模型权重与源码托管。
- **可视化与调试**：TensorBoard 使用、训练过程损失波动分析。

## 使用说明

1. 克隆仓库：

   ```bash
   git clone https://github.com/Zhangxulang/Deep-Learning.git
   cd Deep-Learning
   ```

2. 环境配置：参考`环境搭建.md`配置 CUDA、PyTorch 等依赖。

3. 实战练习：各目录下的`*.ipynb`文件可直接运行，部分脚本需参考对应`md`文档的步骤说明。

## 参考资料

- 论文：《Attention is All You Need》、《LoRA: Low-Rank Adaptation of Large Language Models》等。
- 工具文档：[PyTorch 官方文档](https://pytorch.org/docs/)、[Hugging Face PEFT](https://huggingface.co/docs/peft)。
- 数据集与平台：Kaggle 数据集、Hugging Face Model Hub。
