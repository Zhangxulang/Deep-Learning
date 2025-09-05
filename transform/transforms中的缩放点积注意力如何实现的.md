原论文缩放点积注意力结构如下：



![image-20250818022940664](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250818022940664.png)

# 一、模型结构解读

1. **底层MatMul**

- **输入**：接收Q和 K。Q、K 是经过线性变换后的向量，形状为 `[batch_size, seq_len, d_k]`（`batch_size` 是批次大小，`seq_len` 是序列长度，`d_k` 是每个向量的维度 ）。
- **作用**：计算Q和K之间的相似度。通过矩阵乘法，得到一个分数矩阵，矩阵中的每个元素表示 Q 中某个位置的向量与 K 中各个位置向量的匹配程度，输出形状为 `[batch_size, seq_len, seq_len]` ，即每个位置的Q与所有位置的K的相似度分数。

2. **Scale**

- **输入**：相似度分数矩阵 是上一步矩阵乘法的结果。
- **作用**：对相似度分数进行缩放，缩放因子是 1 / √d_k（`d_k` 是K/Q 向量的维度 ）。这么做是为了缓解因 `d_k` 较大时，点积结果过大，导致SoftMax后梯度消失的问题，让注意力分布更合理。

3. **Mask (opt.)**

- **输入**：接收缩放后的分数矩阵 。

- 作用

  - 在训练语言模型时，对未来的 token 进行掩码（Transformer decoder 中的自注意力 ），确保模型无法看到后续要预测的内容，形状为 `[batch_size, seq_len, seq_len]` 的掩码矩阵，将不需要关注的位置分数设为极小值（如 `-1e9` ），这样经过 SoftMax 后，这些位置的注意力权重就会趋近于 0 。
  - 在处理填充padding的 token 时，掩码可以忽略这些无效 token 的影响 。该模块是可选的，若不需要如文本分类等场景，可跳过。

4. **SoftMax**

- **输入**：接收经过缩放后的分数矩阵 。
- **作用**：对每个查询位置的分数进行归一化，将其转换为注意力权重。经过 SoftMax 后，每行（对应一个查询位置 ）的元素和为 1 ，得到的注意力权重矩阵形状仍为 `[batch_size, seq_len, seq_len]` ，权重值表示每个查询位置对各个键位置的关注程度。

5. **顶层 MatMul**

- **输入**：接收注意力权重矩阵和 V 。V 是经过线性变换后的向量，形状通常为 `[batch_size, seq_len, d_v]`（`d_v` 是 Value 向量的维度，一般和 `d_k` 相关 ）。
- **作用**：通过注意力权重对 Value 进行加权求和。将注意力权重矩阵与 V 进行矩阵乘法，得到最终的注意力输出，形状为 `[batch_size, seq_len, d_v]` ，这个输出融合了根据注意力权重筛选后的 Value 信息，体现了模型对不同位置信息的关注侧重。

# 二、什么是缩放点积注意力？

给定查询 Q、键 K、值 V，缩放点积注意力定义为：
$$
\text{Attention}(Q, K, V) 
= \underbrace{\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{\text{注意力权重}} V
$$




$$
QK^\top 产生相似度/匹配分数
$$



$$
用\sqrt{d_k}做温度缩放再进 softmax
$$




softmax 后得到每个查询对所有键的权重，再与 V 加权求和。

# 三、为什么要“缩放”？

不缩放时，若 Q,K的各维分量近似零均值、单位方差，则
$$
\mathrm{Var}(QK^\top) \propto d_k
$$

$$
\tilde{s}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

d_k越大，分数的绝对值越大，softmax 进入饱和区（极端接近 0 或 1），导致：

- **梯度消失/训练不稳**：softmax 饱和区的梯度很小，反向传播变困难。

- **不同头/层间分布不一致**：改变 d_k 会系统性改变分数尺度。
   用
  $$
  \sqrt{d_k}
  $$
   除一下，就把方差大致归一到常量级，使 softmax 落在更可学习的温度区间，训练更快更稳。这等价于给 softmax 设温度 
  $$
  T=\sqrt{d_k}
  $$
  

# 四、如何实现

单/多头都适用，只是形状不同

1、计算打分矩阵
$$
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}
$$
2、加掩码，把需忽略位置设为 `-inf`
$$
\text{scores}_{ij} =
\begin{cases}
-\infty, & \text{若位置被mask} \\\\
\frac{q_i \cdot k_j}{\sqrt{d_k}}, & \text{否则}
\end{cases}
$$


3、计算注意力权重
$$
\alpha_{ij} = \frac{\exp(\text{scores}_{ij})}{\sum_{j'} \exp(\text{scores}_{ij'})}
$$


4、输出向量
$$
\text{output}_i = \sum_j \alpha_{ij} v_j
$$


多头情形的张量形状

- 输入嵌入 (B,L,d_model)线性映射并重排为
   Q,K,V∈(B,h,L,dk​)，其中为头数，dk=d_model/h。
- 将最后两步在每个头上并行执行，再把 h个头在通道维拼回d_model。

**缩放点积注意力**本质是**点积打分 + 温度缩放 + softmax 加权**；

# 五、项目代码

```py
from dataclasses import dataclass
from typing import Optional, Tuple

Tensor = torch.Tensor

@dataclass
class AttentionOutput:
    hidden_states: Tensor
    attn_scores: Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hyper params
        self.hidden_size = config["d_model"] # 隐藏层大小
        self.num_heads = config["num_heads"] # 多头注意力的头数
        assert (
            self.hidden_size % self.num_heads == 0
        ), "Hidden size must be divisible by num_heads but got {} and {}".format(
            self.hidden_size, self.num_heads
        )
        self.head_dim = self.hidden_size // self.num_heads # 每个头的维度

        # layers
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size, bias=False) #第二个self.hidden_size可以*系数
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # 输出层

    def _split_heads(self, x: Tensor) -> Tensor:
        bs, seq_len, _ = x.shape #假设输入的维度是[batch_size, seq_len, hidden_size],hidden_size是512
        x = x.view(bs, seq_len, self.num_heads, self.head_dim) #num_heads是8，head_dim是64
        return x.permute(0, 2, 1, 3) #变换维度，[batch_size, num_heads, seq_len, head_dim]

    def _merge_heads(self, x: Tensor) -> Tensor:#将多头注意力的输出合并为一个张量
        bs, _, seq_len, _ = x.shape #假设输入的维度是[batch_size, num_heads, seq_len, head_dim]
        return x.permute(0, 2, 1, 3).reshape(bs, seq_len, self.hidden_size) # 变换维度，变为[batch_size, seq_len, hidden_size]

    def forward(self, querys, keys, values, attn_mask=None) -> AttentionOutput:
        # split heads
        querys = self._split_heads(self.Wq(querys)) #(batch_size, seq_len,hidden_dim)-->[batch_size, num_heads, seq_len, head_dim]
        keys = self._split_heads(self.Wk(keys))#[batch_size, num_heads, seq_len, head_dim]
        values = self._split_heads(self.Wv(values))#[batch_size, num_heads, seq_len, head_dim]

        # calculate attention scores
        qk_logits = torch.matmul(querys, keys.mT) # 计算注意力分数，matmul是矩阵乘法，mT是矩阵转置,qk_logits是[batch_size, num_heads, seq_len, seq_len]
        # print(querys.shape[-2], keys.shape[-2])  #3 4
        if attn_mask is not None:
            attn_mask = attn_mask[:, :, : querys.shape[-2], : keys.shape[-2]]
            qk_logits += attn_mask * -1e9 # 给需要mask的地方设置一个负无穷
        attn_scores = F.softmax(qk_logits / (self.head_dim**0.5), dim=-1) # 计算注意力分数

        # apply attention scores
        embeds = torch.matmul(attn_scores, values) # softmax后的结果与value相乘，得到新的表示
        embeds = self.Wo(self._merge_heads(embeds)) # 输出层 [batch_size, seq_len, hidden_size]

        return AttentionOutput(hidden_states=embeds, attn_scores=attn_scores)
```

