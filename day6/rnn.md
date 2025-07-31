# 文本处理

文本分类

文本处理  rnn  循环神经网络



文本比图像消耗GPU更小



 



单热码  多热码

one-hot编码之间没有距离 无法反应近义词



密集向量



embedding



![image-20250731031757342](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250731031757342.png)

情感分类数据集是tensorflow框架自带的

encode

decode  id变文本

### 词典

char level、world level、subword level（大模型）



把文本变为数字

每一个单词映射一个数字

不等长文本映射时需要变为等长的数字个数（选最大的，不足补0）

开始：、结束：、未映射单词：





学习rnn使用

rnn做电影情感分类

![image-20250731041041673](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250731041041673.png)

二进制交叉熵损失：

![image-20250801024453985](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801024453985.png)

单层

双层：前一层的输出作为下一层的输入

双向：越靠前的词信息衰减的越厉害

单向

单层单向、单层双向、双层单向、双层双向



![image-20250801011743669](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801011743669.png)

rnn如何实现文本生成

多输入多输出

RNN输出序列的每一个值，都经过全连接层处理，每个词都输出65个概率值，那个值大就是哪个类别



莎士比亚文集



RNN推理

LSTM 长短期记忆网络   电影情感**分类**

![image-20250801031511060](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801031511060.png)



![image-20250801031917478](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801031917478.png)

![image-20250801032713769](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801032713769.png)



生成

![image-20250801033035433](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801033035433.png)

用subword的目的就是减少词典数量，避免词典出现UNK

subword-nmt工具比jieba支持语言更多

数据集如下：

![image-20250801033535552](C:\Users\oyaZXL\AppData\Roaming\Typora\typora-user-images\image-20250801033535552.png)

数据集网址：https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/discussion?sort=hotness



LSTM做电影情感分类  subword分词

Seq2Seq 序列到序列





