# GPT-1

只用decoder，12层，参数上亿
为了解决不同类型的问题，使用同一个transforms，最后输出层不同



# BERT

只用encoder模块

大量的五标签的数据集上，【mask】训练

Large模型3.4亿参数

# GPT-2

只用decoder、one-hot、15亿参数

需要涉及各种各样的问题模板，去训练模型（元学习）

能够解决各种各样的问题

# T5

Bean Search   束搜索

跨度掩码    挖了多个mask，这些mask不连续

海里数据集、110亿

# GPT-3

prompt

提问  zero-hot、one-hot、few-hot

数据集清洗  二分类模型、按概率值排序（可以从低质量的数据集中提取质量还不错的）、LSH局部敏感哈希（数据集已经很多了，需要把很相似的样本删除）

创新点：

更大数据集

LSH（局部敏感哈希）算法

 强化学习RL模型

对齐

# GPT-3.5即ChatGPT

解决问题

相同的的模型结构构造一个尺寸为60亿RL模型   40名标注工（对模型输出排序，不同排名设一个分数）

通过PPO（近端策略优化）微调1750亿参数的大模型    全量微调

训练RL模型，来教ChatGPT

# GPT-5



beam searc