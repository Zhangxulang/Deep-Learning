### GPU设置

单机多卡

多GPU训练策略

### 机器学习

tutorial





完成Seq2Seq实战

1. preprocessing data ---数据id化和dataset生成

实现词典

Tokenizer **word level Tokenizer**

DataLoader

2. build model

 2.1 encoder构建（使用GRU）

2.2 attention构建

实现Bahdanau----重点，难点

2.3 decoder构建

用的lstm变种GRU

 2.4 loss& optimizer

自定义梯度的更新

2.5 train 每次epoch调用train

3. evaluation（不适合看准确率）  bleu

 3.1 given sentence, return translated results

3.2 visualize results (attention) 

 

# **第一节课**

***\*Tokenizer\****  ***\*word level\****

## **DataLoader  多个tensor**

![image-20250802172155722](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250802172155722.png)

![image-20250802172457321](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250802172457321.png)

翻译出的结果  today is a good day

decoder_inputs  [BOS] today is a good day

decoder_labels  today is a good day [EOS]

 

\# ***\*2. build model\****

\# 2.1 encoder构建（使用GRU）

 



# **第二节课**

\# 2. build model

\# 2.1 encoder构建（使用GRU）

\# 2.2 attention构建

***\*实现\*******\*Bahdanau\**** ***\*----重点，难点\****

***\*加性注意力\****

 

\# 2.3 decoder构建

用的lstm变种GRU

 score=FC(tanh(FC(EO)+FC(H)))

context = sum(attention_weights * EO, axis = 1) 

attention_weights = softmax(score, axis = 1)

2.4 squence2squence

# **第三节课**

\# 2.5 decoder构建

\# 2.6 Sequence2Sequence

 

Decoder_input  （batch_size,1）

(logits)Decoder_output  （batch_size,1,vocab_size）

 

(batch_size,seq_length,vocab_size)

 

cline VScode  deepseek  (做一个免费的cursor，copilot)

 

# **第四节课**

\# 2.5 loss& optimizer

自定义梯度的更新

\# 2.6 train 每次epoch调用***\*train\****

 

***\*# 3. evaluation\**** ***\*（不适合看准确率）  bleu\****

\# 3.1 given sentence, return translated results

\# 3.2 visualize results (attention) 

 

