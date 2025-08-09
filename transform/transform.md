## **背景**

抛弃循环神经网络RNN---->循环神经网络解决的是序列问题，可以让网络感知到词的前后关系--->**位置编码**

 

## **模型结构**

how  are you [EOS]

 

[BOS] how are you

 

## **位置编码**

Embeds  (batch_size,seq_len,embed_dim)

位置编码尺寸和embeds是一致的

 

缩放点积注意力公式

 

# **第三节课**

缩放点积注意力公式  形象举例

(seq_len,hidden_dim)

A,B,C,D分别是序列里的四个词

 

为什么除以根号dk

q，k，v怎么来

 

多头注意力原理

 

位置编码公式 

 

 

# **第四节课**

层归一化原理

层归一化与批归一化区别

 

学习率变化设计

 

实战步骤

sh是linux可以运行的脚本文件

 

德语和英语，使用的一个词典 Tokenizer

 

sys.argv  







**实战步骤： ----这是实战代码的一个大纲**

 

\# 1. loads data，data_multi30k.sh 处理(和 seq2seq 类似，区别是 subword) 原始数据文件是

![img](file:///C:\Windows\Temp\ksohtml11392\wps2.png) 

data_multi30k.py   分词 ，还有变小写 ，去空格

 

 

\# 2. DataLoader 准备 LangPairDataset

Tokenizer

 

 

TokenBatchCreator

TransformerBatchSampler（可以最佳的利用 GPU，动态的 batch_size）

 

\# 3. TransformerEmbedding

完成普通 embedding，***\*位置编码\****，并将其相加

 

\# 4. builds model 分为以下 6 步

***\*王道码农训练营\*******\*-WWW.CSKAOYAN.COM\****



\# 4.1 MultiheadAttention  --***\*难点，最难\****

 

\# 4.2 TransformerBlock  --***\*难点\****  

\# 4.3 TransformerEncoder

\# 4.4 TransformerDecoder

\# 4.5 generate_square_subsequent_mask --***\*难点\****

\# 4.6 TransformerModel

***\*TransformerModel\**** ***\*就是把\**** ***\*Encoder\**** ***\*和\**** ***\*Decoder\**** ***\*组合起来\****

\# 5. CrossEntropyWithPadding（损失函数）& NoamDecayScheduler（学习率衰 减）

\# 6. train step -> train

\# 7. Evaluate and Visualize











