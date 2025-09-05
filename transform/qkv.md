多头注意力通过 “拆分 - 并行计算 - 拼接 - 融合” 的流程

输入到多头注意力的维度是d_model=512(一个样本有512个特征token)

一、输入一个样本，样本长度为seq_len,拆分为一个一个的token

二、token转为id

三、id转向量，得到输入矩阵x   （seq_len  x  d_model）      **这一步由class TransformerEmbedding(nn.Module):实现**

四、并行计算：每一个向量在每一个头上分别计算q，k，v=W_q* x,W_k* x, W_v *x(逐步求特征矩阵W_q ,W_k , W_v ，每个矩阵维度`512×512`)   ；q，k，v维度是seq_len×512

四、**拆分多头**
将`Q、K、V`按特征维度拆分为`num_heads=8`个并行的子矩阵，每个子矩阵的特征维度为`d_k = 512/8 = 64`：

- 每个头的`Q_i、K_i、V_i`形状为`(seq_len, d_k)`（即`6×64`）
- 拆分后整体形状变为`(num_heads, seq_len, d_k)`（即`8×6×64`）

五、拼接：(batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, d_model)

六、融合输出（维度保持d_model）：w_o





### 修订流程修订：输入到多头注意力的完整流程（基于 d_model=512）

#### 一、输入序列预处理

1. **原始输入**：一个样本是长度为`seq_len`的文本序列（如一句话），例如：`"The cat sits on the mat"`（假设`seq_len=6`）。

2. **拆分 token**：将文本拆分为`seq_len`个 token（如单词或子词），得到：`[The, cat, sits, on, the, mat]`。

3. **token 转 id**：通过词汇表将每个 token 映射为整数 id，得到形状为`(seq_len,)`的 id 序列，例如：`[3, 15, 42, 7, 3, 58]`。

4. id 转向量（嵌入层）

   ：通过嵌入矩阵（维度

   ```
   [vocab_size, d_model]
   ```

   ）将 id 转为向量，得到输入矩阵

   ```
   x
   ```

   ，形状为

   ```
   (seq_len, d_model)
   ```

   （即

   ```
   6×512
   ```

   ）。

   - 每个 token 对应一个 512 维的向量，整个序列形成`seq_len×512`的矩阵（非单个 token 有 512 个特征）。

#### 二、多头注意力计算

1. **线性变换生成 Q、K、V**
   对输入矩阵`x`（`seq_len×512`）分别应用 3 个可学习的线性变换矩阵`W_q、W_k、W_v`（每个矩阵维度`512×512`），得到：
   - 查询矩阵`Q = x × W_q`（形状`seq_len×512`）
   - 键矩阵`K = x × W_k`（形状`seq_len×512`）
   - 值矩阵`V = x × W_v`（形状`seq_len×512`）
     （注：这里是矩阵乘法，而非逐 token 独立计算）
2. **拆分多头**
   将`Q、K、V`按特征维度拆分为`num_heads=8`个并行的子矩阵，每个子矩阵的特征维度为`d_k = 512/8 = 64`：
   - 每个头的`Q_i、K_i、V_i`形状为`(seq_len, d_k)`（即`6×64`）
   - 拆分后整体形状变为`(num_heads, seq_len, d_k)`（即`8×6×64`）
3. **并行计算缩放点积注意力**
   每个头独立计算注意力输出：
   - 步骤 1：计算注意力分数（点积）：`scores_i = Q_i × K_i^T`（形状`6×6`，表示每个 token 对其他 token 的关联度）
   - 步骤 2：缩放：`scores_i = scores_i / √d_k`（√64=8，防止分数过大）
   - 步骤 3：归一化：`attn_weights_i = softmax(scores_i)`（形状`6×6`，权重和为 1）
   - 步骤 4：加权求和：`head_i = attn_weights_i × V_i`（形状`6×64`，每个 token 的输出向量）
4. **拼接多头输出**
   将 8 个头的输出`head_1`到`head_8`（每个`6×64`）在特征维度拼接，得到形状为`(seq_len, d_model)`的矩阵（`6×512`，因`8×64=512`）。
5. **融合输出（线性变换）**
   对拼接后的矩阵应用可学习的线性变换矩阵`W_o`（维度`512×512`），得到最终输出`output`，形状仍为`(seq_len, d_model)`（`6×512`），与输入矩阵`x`维度一致。

#### 三、残差连接与层归一化（补充步骤）

1. **残差连接**：`output = x + output`（输入与输出逐元素相加，因维度一致）。
2. **层归一化**：对结果进行层归一化，稳定训练：`output = LayerNorm(output)`。

### 关键修正说明

1. **维度概念澄清**：
   - `d_model=512`是每个 token 的特征维度（每个 token 用 512 维向量表示），而非 “512 个特征 token”。
   - 序列长度`seq_len`是 token 的数量（如 6 个单词），与`d_model`无关。
2. **矩阵运算修正**：
   - `Q、K、V`是通过矩阵乘法批量计算的（整个序列一次性处理），而非 “每个向量在每个头上分别计算”。
   - 线性变换矩阵`W_q、W_k、W_v`是全局共享的，而非 “逐步求特征矩阵”。
3. **流程完整性**：
   补充了注意力分数计算的细节（缩放、softmax）和残差连接步骤，这是多头注意力在 Transformer 中的完整应用流程。

# tranform中什么是查询（Query）、键（Key）、值（Value）

想象你去图书馆借书，把整个过程拆成三步就能秒懂 Q、K、V：

1. 查询（Query）＝你写在纸条上的问题
   “我想找一本讲狗的书。”
   这张纸条就是 Query，告诉别人你现在想要什么。
2. 键（Key）＝每本书封面上的标签
   图书馆给每本书都贴了一个标签：
   “猫”“狗”“恐龙”“烹饪”……
   这些标签就是 Key，用来快速匹配你的问题。
3. 值（Value）＝书里的实际内容
   真正被你带回家阅读的文字、图片、知识，就是 Value。

**Transformer 做的事，其实就是放大版的“图书馆找书”：**

- 把 Query 和所有 Key 拿去“对对碰”（算相似度）。
- 相似度越高，说明这本书（Key）越符合你的问题（Query），就给它打高分。
- 最后把高分的书（Value）按比例拼在一起，生成最终的回答。

一句话总结：
**Query 是你要啥，Key 是标签匹配，Value 是实际给到手的知识。**



# 那 Transformer 怎么知道哪个 Key 最匹配 Query？

用“打分”这一招，简单又粗暴。

1. 把 Query 和每一个 Key 都变成同样长度的向量（一串数字）。
2. 计算“点积”——也就是把两段数字对应相乘再相加，得到一个分数。
   - 分数高＝方向接近＝很匹配。
   - 分数低＝方向差得远＝不匹配。
3. 为了让分数变成“百分比”，再做一次 softmax：
   - 最高分被放大，低分被压扁，所有分数加起来正好等于 1。
4. 用这些百分比去“加权”对应的 Value，最后把加权的 Value 相加，就得到输出。

形象一点：
把 Query 想成磁铁，Key 想成小铁钉，磁铁对每个铁钉的“吸力”就是分数；吸力越强，对应的 Value 被“吸”过来的比例就越大。Transformer 根本不用“懂”内容，只要算数字之间的距离就能知道谁最匹配。

# 查询（Query）、键（Key）、值（Value）是怎么来的

它们不是天上掉下来的，而是“同一句话”在 Transformer 里被投影成了三种身份。

1. 先把输入的每个词（或字）变成向量
   例如把“狗”变成 512 个数字：x。
2. 再准备三套可学习的“投影灯”——三个权重矩阵
   W_Q、W_K、W_V（名字就叫 Q、K、V 的权重矩阵）。
3. 把同一个向量 x 分别照进这三盏灯，就得到三种身份
   Q = x · W_Q  （拿去提问）
   K = x · W_K  （拿去当标签）
   V = x · W_V  （拿去当内容）

所以一句话里的每个词，都会同时生成自己的 Q、K、V。
训练时，模型通过不断调整 W_Q、W_K、W_V 的参数，学会“怎样把信息拆成提问、标签、内容”才最有利于完成任务。

# 把输入的每个词（或字）变成向量是用的wordtoid技术吗

不是。
Word2Vec（包括 CBOW、Skip-gram、GloVe 等）只是早期把“词→向量”的一种静态方法，Transformer 里用的是更通用的两步：

1. Token → ID
   先把词或子词（sub-word）切成 token，再用词表把它映射成一个整数 ID。这一步只是查字典，跟 Word2Vec 没直接关系。
2. ID → 向量
   把 ID 再查一张“可训练的嵌入表”（embedding table，本质是一个矩阵 E），得到固定维度的向量。这张表是 Transformer 自己的参数，一开始随机初始化，然后通过任务一起训练，而不是预先拿 Word2Vec 训好再冻结。

因此，Transformer 的词向量是“任务导向、端到端训练”的嵌入，而不是“预先用 Word2Vec 算好、再搬进来”的静态向量。

# ID → 向量具体是如何实现的

一句话：查表 + 可选的位置编码。
下面把「ID → 向量」拆成 3 个具体步骤，让你看到代码长什么样，也看到数据怎么变。

------

1. 准备一张「可训练表」
   设词表大小 V=30 000，嵌入维度 d=512。
   随机初始化一个矩阵
   E ∈ ℝ^{V×d}（30 000 行，512 列）
   每一行就是某个 token-ID 对应的 512 维向量。
2. 查表：把整数 ID 换成向量
   假设输入一句话被 tokenizer 切成 5 个 token，得到 ID 序列
   ids = [17, 5, 800, 3, 9]
   在矩阵 E 里把这 5 行一次性抽出来：
   X = E[ids]            # 形状变成 (5, 512)
   这 5 个 512 维向量就是最初的「词嵌入」。
3. 加上位置信息（Positional Encoding）
   因为 Transformer 没有循环/卷积，需要显式告诉它每个词的位置。
   做法：再生成一个同样形状的位置矩阵 P ∈ ℝ^{5×512}，按论文里的 sin/cos 公式计算，或者干脆再训练一个表。
   相加即可：
   X = X + P              # 仍然是 (5, 512)

到此，「ID → 向量」完成，X 被送进后续的 Q/K/V 投影层。

------

PyTorch 伪代码（一行就能完成步骤 2

```python
import torch.nn as nn

embed = nn.Embedding(num_embeddings=30000, embedding_dim=512)
x = embed(torch.tensor([17, 5, 800, 3, 9]))   # 输出 (5, 512)
```

就这么简单：矩阵查表 + 可选位置编码。

# X 被送进后续的 Q/K/V 投影层具体是如何实现的

一句话：还是“再查三张可训练的小表”——把同一个 X 分别乘上 W_Q、W_K、W_V 三个矩阵，就得到 Q、K、V。

下面用 4 步把它讲透，并给出 PyTorch 代码。

------

1. 拿到上一步的 X
   形状 (seq_len, d_model)
   例：seq_len=5，d_model=512 → (5, 512)

2. 准备 3 个投影矩阵
   W_Q, W_K, W_V ∈ ℝ^{d_model × d_k}
   通常 d_k = d_model / num_heads
   假设 8 个头，则 d_k = 512/8 = 64
   这三个矩阵也是随机初始化、一起训练的参数。

3. 做 3 次线性变换（就是矩阵乘法）
   Q = X · W_Q      # (5,512) × (512,64) → (5,64)
   K = X · W_K      # 同上
   V = X · W_V      # 同上

   这样就得到三个形状完全相同的新张量：Q、K、V。

4. （多头时）再 reshape
   把 Q、K、V 从 (5,64) 改成 (5,8,8) 再转置 → (8,5,8)
   每个头独立算注意力，最后再拼回来。
   这一步只是为了让“多个注意力头”并行，不改变本质。

------

PyTorch 一行就能完成步骤 2 和 3：

```python
import torch.nn as nn

# 假设 d_model=512, nhead=8 → d_k=d_v=64
linear_q = nn.Linear(512, 64)
linear_k = nn.Linear(512, 64)
linear_v = nn.Linear(512, 64)

Q = linear_q(X)  # (5, 64)
K = linear_k(X)  # (5, 64)
V = linear_v(X)  # (5, 64)
```

如果你用官方 `nn.MultiheadAttention`，它内部已经封装了这三行。

总结：
“投影层”就是再学 3 个权重矩阵，把同一个嵌入向量线性映射成 3 份，分别当 Query、Key、Value。



# 把 Q、K、V 从 (5,64) 改成 (5,8,8) 再转置 → (8,5,8)为什么要这么做

一句话：为了让 8 个“小脑袋”能并行地各自算注意力，最后再拼成一个整体回答，既加能力又不增加计算量。

把 (5,64) 拆成 (8,5,8) 的每一步到底在干嘛？下面用“搬砖”的比喻拆开说：

1. 64 维是 8 个头的“工资总额”
   原来 64 维里既装狗的语义，又装颜色、大小等信息，混在一起。
   现在把它平均切成 8 份，每份 8 维，相当于给 8 个工人各发一份工资，让他们专注不同的子任务。
2. 把顺序从 (seq_len, d_k_total) 变成 (nhead, seq_len, d_k)
   • (5,64) → (5,8,8)：先切成 8 段，每段 8 维，但还是“按词排班”。
   • 再转置 → (8,5,8)：现在变成“按头排班”，同一头的 5 个词挨在一起。
   这样 GPU 可以一次性把 8 份 5×8 的小矩阵并行乘完，而不是在 64 维大矩阵里串行算 8 次。
3. 并行算注意力
   每个头拿自己的 8 维 Q、K、V 做缩放点积注意力，得到 8 份 5×8 的输出。
   8 个头各自捕获不同的关系（如语法、指代、情感等），互不干扰。
4. 最后再拼回去
   把 8 份 (5,8) 按最后一维拼接，变回 (5,64)，再乘一个输出矩阵 → (5,512)。
   相当于让 8 个工人把各自的“局部报告”汇总成一份完整答案。

一句话：
(5,64)→(8,5,8) 的 reshape+transpose 只是为了让“多头并行计算”在代码里变得高效、直观，并不改变数学意义，只是把 64 维切成 8 条 8 维的小通道。

8 个头算得是同一个句子（样本）的注意力分数还是同一个句子的不同部分的注意力分数

复制

8 个头算的是**同一个句子的注意力分数**，但各自“看”句子的**不同子空间（不同部分的信息）**，而不是把句子切成 8 段。

用一句话比喻：
把同一张照片复印 8 份，交给 8 位画家，每人只用一种颜色的笔在复印纸上画细节；最后把 8 张局部图叠在一起，就得到一张更丰富的完整画作。

具体细节：
• 输入始终是整句话（5 个 token）。
• 每个头拿到的是**整句话**在 8 维子空间里的 Q、K、V。
• 每个头都会为**所有 5×5=25 个位置对**计算注意力分数，只是关注的是不同“语义维度”。
• 8 个头并行算完后，把 8 份 5×8 的输出在最后一维拼接，再映射回原维度；信息融合，句子仍是完整的。

# 8 个头并行算完后得到的是什么

8 个头各自算完后，手里拿的是 8 份 **「加权后的子空间表示」**，形状都是
(5, 8) —— 5 个 token，每个 token 在该头对应的 8 维子空间里被重新表示了一次。

把这 8 份 (5, 8) 在第 2 维（头维）上 **拼接** 起来，就得到
(5, 64) —— 也就是把 8 个 8 维向量首尾相接，还原成原来的 64 维。

最后再乘一个输出矩阵 W_O（64 × d_model，通常 d_model=512），得到
(5, 512) —— 这才是进入下一层 Transformer 的「多头注意力输出」。

# 为什么要乘一个输出矩阵 W_O

把 8 个头的结果“粘”在一起后，只是简单地把 8 条 8 维向量拼成一条 64 维向量，这样做有两个问题：

1. 信息仅仅是“堆”在一起，没有让不同头之间再交流、再整合。
2. 维度通常比模型内部真正需要的 d_model 要小（64 ≠ 512），需要再映射回去。

乘一个可学习的输出矩阵 W_O（64×512）就是：

• 做一次线性变换，让 64 维的拼接结果重新投影回模型的标准维度 512；
• 同时给模型一个“混合”机会——每个输出通道都可能是 8 个头里任意维度的加权和，相当于让各头信息再次融合；
• 还能引入新的非线性表达能力（后面一般会接残差 + LayerNorm + FFN）。

一句话：W_O 是“把多头结果重新打包成统一格式，并再做一次可学习的整合”。

# W_O 矩阵如何设计？

一句话：W_O 就是一个普通的线性层，形状固定为
(heads × head_dim) × d_model，
权重随机初始化，跟着整个 Transformer 一起用梯度下降训练，不需要也不应该人为设计数值。

具体细节

1. 形状
   heads = 8（举例）
   head_dim = 64
   d_model = 512
   ⇒ W_O ∈ ℝ^{(8×64) × 512} = ℝ^{512×512}
2. 初始化
   用 Xavier/Kaiming 均匀或正态分布初始化即可，和 Transformer 里所有线性层一样。
3. 训练
   与 Query、Key、Value、FFN 等所有参数一起端到端更新，没有任何特殊约束或人工设定规则。
4. 推理
   训练完成后权重固定，直接矩阵乘法：
   MultiHeadOut = Concat(head1,…,head8) · W_O

因此，W_O 的“设计”只有两步：选对形状 + 标准初始化，其余交给模型自己学。

# 每个token都会同时生成自己的 Q、K、V吗

是的。
在 Transformer 的自注意力里：

1. 先把整句话的每个 token 都通过同一张「词嵌入表」得到向量 x₁, x₂, …, xₙ（n 就是序列长度）。
2. 接着 **每个 token 的向量 xᵢ 分别乘上同一组 W_Q、W_K、W_V** 三个矩阵。
   - Qᵢ = xᵢ · W_Q
   - Kᵢ = xᵢ · W_K
   - Vᵢ = xᵢ · W_V

于是，一句话里 **每个 token 都会同时拥有自己的一份 Q、K、V**，形状都是 (d_k,)，或者多头情况下 (heads, head_dim)。
这些 Q/K/V 再拿去跟句子中所有其它 token 的 K/V 做注意力计算。

# 注意力机制中，token 如何与其他 token 进行交互？

一句话：每个 token 用自己生成的 **Q** 去“查”所有 token（包括自己）生成的 **K**，再用得到的权重去“取”所有 token 的 **V**，于是所有 token 的信息就被加权融进了当前 token 的新表示。

下面把全过程拆成 4 步，用一句话里的 3 个 token 做例子：

1. 各自备好 Q、K、V
   设句子长度为 3，d_k=4：
   token₁：(Q₁, K₁, V₁)
   token₂：(Q₂, K₂, V₂)
   token₃：(Q₃, K₃, V₃)
2. 计算注意力分数（打分）
   拿 token₁ 举例：
   score₁₁ = Q₁·K₁
   score₁₂ = Q₁·K₂
   score₁₃ = Q₁·K₃
   得到一个 3 维向量 [s₁₁, s₁₂, s₁₃]。
   对所有 token 都这么做，就得到 3×3 的分数矩阵。
3. 转成权重（概率）
   对分数做 softmax，保证每行和为 1：
   weight₁ = softmax([s₁₁, s₁₂, s₁₃])
   现在 token₁ 与每个 token 的“亲密度”用 3 个小数表示。
4. 加权求和得到新的 token 表示
   new_token₁ = weight₁₁·V₁ + weight₁₂·V₂ + weight₁₃·V₃
   同样地，new_token₂、new_token₃ 也按各自的权重把全句 V 加权求和。

结果：每个 token 的新向量都“看见”了整句的所有信息，但看多看少由注意力权重决定。多头只是把上述过程在 8 个不同的子空间里并行再做 8 次，最后把 8 份结果拼接并线性映射回原始维度。

# 从输入token到多头注意力输出，详细介绍都经历了哪些层，做了什么运算，为什么怎么做

下面以一条长度为 seq_len 的输入句子为例，按时间轴把「从原始 token 到多头注意力输出」完整拆开。
所有张量尺寸都用常见配置举例：
词表大小 30 000，d_model=512，n_head=8，head_dim=64，seq_len=5。

────────────────

1. Token → ID（查字典）
   原始句子 → 子词切分 → [17, 5, 800, 3, 9]
   形状：(5,) 的 int64 向量。
2. ID → 词向量（查表 + 位置编码）
   • nn.Embedding(30000, 512) 把每个 ID 换成 512 维向量：
   X_emb = E[ids] → (5, 512)
   • 加上位置编码 P（sin/cos 或学习得到）：
   X = X_emb + P           # 仍是 (5, 512)
   目的：给模型顺序信息。
3. LayerNorm（预-LN 结构）
   X = LayerNorm(X)          # (5, 512)
   目的：稳定梯度，使后续矩阵乘法数值尺度一致。
4. 线性投影 → 拿到 Q、K、V
   • 三个权重矩阵 W_Q, W_K, W_V ∈ ℝ^{512×512}
   • 先整体算出 Q、K、V：
   Q = X·W_Q = (5,512)×(512,512) → (5,512)
   K = X·W_K → (5,512)
   V = X·W_V → (5,512)
   目的：把同一套输入映射到三种不同语义空间。
5. Reshape & Transpose → 多头形态
   • 把 512 维切成 8 段 64 维：
   Q → (5,8,64)
   • 转置 → (8,5,64)（同理 K、V）
   目的：让 8 个头并行计算，每个头关注 64 维子空间。
6. 缩放点积注意力（每个头内部）
   对单头（形状(5,64)）说明：
   a) 分数 S = Q·K^T         # (5,5)
   b) 缩放 S = S / √64       # 防止 softmax 饱和
   c) Mask（可选）           # 解码器用，防止看到未来
   d) softmax → 权重矩阵 A   # 行和为1
   e) 加权求和 Out = A·V     # (5,64)
   8 个头同时算完，得到 8 份 (5,64)。
7. Concat → 还原单矩阵
   把 8 个 (5,64) 在第 2 维拼接 → (5,512)。
8. 输出投影 W_O
   Out = Concat(heads)·W_O   # W_O ∈ ℝ^{512×512}
   目的：
   • 把 512 拼接向量再线性变换回 512，让各头信息充分混合；
   • 维持输入输出维度一致，便于残差相加。
9. 残差连接
   Z = X + Out               # 仍是 (5,512)
   目的：保留原始信号，缓解梯度消失。
10. 再次 LayerNorm（可选，取决于实现）
    有些实现把 LN 放在残差之后，得到最终多头注意力输出。

────────────────
一句话串起来：
token → 查表+位置 → LayerNorm → 线性映射得 QKV → 拆头并行算注意力 → 合并头 → 再线性 → 残差 → LayerNorm → 得到形状不变的 (5,512) 向量，继续喂给后面的 FFN。

每一部分都给出代码完整例子，要求可验证输入输出

下面给出一段**可运行、可验证**的完整 PyTorch 代码，从「原始句子」到「多头注意力输出」每一步都打印张量形状与数值，确保完全透明。
依赖：PyTorch ≥1.9（CPU 即可）。



```bash
pip install torch
```



```python
import torch
import torch.nn as nn
import math

# 1. 超参数
vocab_size   = 30_000
d_model      = 512
n_head       = 8
head_dim     = d_model // n_head
seq_len      = 5
batch        = 1          # 演示用单句

# 2. 随机造一句 token id（可改）
token_ids = torch.randint(0, vocab_size, (batch, seq_len))  # (1,5)
print("token_ids shape:", token_ids.shape)
print("token_ids:", token_ids)

# 3. 词嵌入 + 位置编码
embed = nn.Embedding(vocab_size, d_model)

# 手工 sin/cos 位置编码（可学习也可）
pos_enc = torch.zeros(seq_len, d_model)
pos      = torch.arange(0, seq_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
pos_enc[:, 0::2] = torch.sin(pos * div_term)
pos_enc[:, 1::2] = torch.cos(pos * div_term)
pos_enc = pos_enc.unsqueeze(0)  # (1,5,512)

x = embed(token_ids) + pos_enc   # (1,5,512)
print("\n嵌入+位置编码 x shape:", x.shape)

# 4. LayerNorm（预归一化）
ln = nn.LayerNorm(d_model)
x = ln(x)

# 5. 产生 Q,K,V（一次性 512→512，再拆多头）
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(x)  # (1,5,512)
K = W_k(x)
V = W_v(x)

# 6. 拆多头 (batch, seq, d_model) -> (batch, n_head, seq, head_dim)
def split_heads(tensor):
    B, L, _ = tensor.size()
    return tensor.view(B, L, n_head, head_dim).transpose(1, 2)

Q = split_heads(Q)  # (1,8,5,64)
K = split_heads(K)
V = split_heads(V)

print("\nQ 拆头后 shape:", Q.shape)

# 7. 缩放点积注意力
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)  # (1,8,5,5)
attn_weights = torch.softmax(scores, dim=-1)                         # (1,8,5,5)
out = torch.matmul(attn_weights, V)                                  # (1,8,5,64)

# 8. 合并多头
def combine_heads(tensor):
    B, H, L, D = tensor.size()
    return tensor.transpose(1, 2).contiguous().view(B, L, H*D)

out = combine_heads(out)  # (1,5,512)
print("\n合并头后 out shape:", out.shape)

# 9. 输出投影 W_O
W_o = nn.Linear(d_model, d_model)
out = W_o(out)            # (1,5,512)

# 10. 残差 + 再次 LayerNorm（可选）
out = ln(x + out)
print("\n最终多头注意力输出 shape:", out.shape)
print("前 5 个值（token0 的前 5 维）:", out[0,0,:5])
```

运行效果（每次数值随机，但形状固定）：



复制

```
token_ids shape: torch.Size([1, 5])
token_ids: tensor([[2345,  123,  9876,   42, 15000]])

嵌入+位置编码 x shape: torch.Size([1, 5, 512])

Q 拆头后 shape: torch.Size([1, 8, 5, 64])

合并头后 out shape: torch.Size([1, 5, 512])

最终多头注意力输出 shape: torch.Size([1, 5, 512])
前 5 个值（token0 的前 5 维）: tensor([...], grad_fn=<SliceBackward0>)
```

至此，从 token id 到多头注意力输出的每一步都已完成，且可直接复制运行验证。