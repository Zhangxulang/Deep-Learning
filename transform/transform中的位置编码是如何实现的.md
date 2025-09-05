# 一、什么是位置编码？

在**Transformer**中，模型的输入是一个序列。
每个词首先会被转换成一个向量。

- Transformer 的核心是**自注意力机制**。
- 注意力机制本身只关心“哪些词与哪些词相关”，却**不关心词在序列中的先后顺序**。
- 这就导致，如果我们只输入 embedding，模型完全分不清 **“我爱你”** 和 **“你爱我”** 的区别。

为了让模型能够感知**序列中单词的位置**，我们必须给每个 embedding 加入“位置信息”,这就是**位置编码**。位置编码的维度与embedding相同，因此两者可以相加。

# 二、为什么需要位置编码？

**核心原因：**

- 自注意力机制是**全连接**的，它认为句子里的词是一个集合（无序）。
- 语言是有顺序的，缺少顺序信息会导致模型无法理解语义。

- “猫追狗”和“狗追猫”，没有顺序感时，注意力计算的结果是一模一样的。
- 有了位置编码后，模型知道“猫”在前，“狗”在后，从而区分不同的语义。

# 三、如何实现？

 Transformer原论文《Attention Is All You Need》采用的方法公式如下（d_model 表示向量维度，pos 表示位置）：
$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{\tfrac{2i}{d_{\text{model}}}}}\right)
$$

$$
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{\tfrac{2i}{d_{\text{model}}}}}\right)
$$

- 每个位置 `pos` 会生成一个维度为 `d_model` 的向量。
- 偶数维用 `sin`，奇数维用 `cos`。
- 不同维度对应不同频率，使得模型可以捕捉到长短不同的依赖关系。
- 这种编码是**固定的、无需训练**。

相当于用“不同频率的波”来表示位置信息，保证不同位置的编码彼此区分，同时相对位置关系也能被表达。

实现细节如下：

```python
def get_positional_encoding(self, max_length, hidden_size):#max_length表示句子里最多有多少token，hidden_size与embedding维度相等，也就是每个 token 向量的大小
    pe = torch.zeros(max_length, hidden_size) # 初始化位置编码,全为0
    # .unsqueeze(1)在张量的维度上增加了一个维度，使其从一维变为二维，第二维的大小为 1。这样做为了后面矩阵广播，方便和 div_term相乘
    position = torch.arange(0, max_length).unsqueeze(1) # 位置信息,从0到max_length-1
    div_term = torch.exp(torch.arange(0, hidden_size, 2)* -(torch.log(torch.Tensor([10000.0])) / hidden_size)
    )# 计算位置编码的权重
     #利用广播机制，得到一个 (max_length, hidden_size/2) 的矩阵，每个元素是 pos/ (10000^(2i/d))
    pe[:, 0::2] = torch.sin(position * div_term)  
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

设常数为c,则
$$
c= -\frac{\ln 10000}{d_model}
$$
利用指数与对数关系，有
$$
div-term=e^{(2i)c} \;=\; e^{-\tfrac{2i}{d_model}\ln 10000}
\;=\; 10000^{-\tfrac{2i}{d_model}}
\;=\; \frac{1}{10000^{2i/d_model}}
$$
然后在奇数位置和偶数位置分别填入torch.cos(position * div_term)和torch.sin(position * div_term) 得到**pe**



最终，**pe**会加到词向量 embedding 上一起作为 Transformer 编码器或解码器的输入
$$
xinput=Embedding(tokens)+PositionalEncoding(positions)
$$
这种相加操作将位置信息与语义信息融合，使得模型在处理每个词时，不仅能获取到词本身的语义表示，还能获取到该词在序列中的位置信息，从而有效注入序列顺序信息，帮助模型更好地理解文本的顺序结构。