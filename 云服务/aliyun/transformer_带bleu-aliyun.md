```python
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

print(sys.version_info)
for module in mpl, np, pd, sklearn, torch:
    print(module.__name__, module.__version__)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

```

    sys.version_info(major=3, minor=12, micro=1, releaselevel='final', serial=0)
    matplotlib 3.8.3
    numpy 1.26.4
    pandas 2.3.1
    sklearn 1.7.1
    torch 2.2.1+cpu
    cpu



```python
#挂载谷歌云盘

# from google.colab import drive
# drive.mount('/content/drive')
```


```python
# !cp /content/drive/MyDrive/transformer-de-en/* . -r
```


```python
# !rm -rf wmt16_cut/
```

## 数据加载

- 采用WMT16的德语和英语平行语料库，数据集主页：[WMT16](https://www.statmt.org/wmt16/multimodal-task.html#task1)


```python
#和jieba分词类似
# !pip install sacremoses
# !pip install subword-nmt
# # BPE分词

```


```python
!pwd
```

    /mnt/workspace/transform



```python
# !sh data_multi30k.sh wmt16 de en
```

Dataset


```python
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class LangPairDataset(Dataset):

    def __init__(
        self, mode="train", max_length=128, overwrite_cache=False, data_dir="wmt16",
    ):
        self.data_dir = Path(data_dir)
        cache_path = self.data_dir / ".cache" / f"de2en_{mode}_{max_length}.npy"

        if overwrite_cache or not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.data_dir / f"{mode}_src.bpe", "r", encoding="utf8") as file:
                self.src = file.readlines() # 读取源语言文件所有行

            with open(self.data_dir / f"{mode}_trg.bpe", "r", encoding="utf8") as file:
                self.trg = file.readlines() # 读取目标语言文件所有行

            filtered_src = []
            filtered_trg = []
            # max length filter,超出最大长度的句子舍弃
            for src, trg in zip(self.src, self.trg):
                if len(src) <= max_length and len(trg) <= max_length: # 过滤长度超过最大长度的句子
                    filtered_src.append(src.strip()) # 去掉句子前后的空格
                    filtered_trg.append(trg.strip())
            filtered_src = np.array(filtered_src)
            filtered_trg = np.array(filtered_trg)
            np.save(
                cache_path,
                {"src": filtered_src, "trg": filtered_trg },
                allow_pickle=True,
            )#allow_pickle=True允许保存对象数组，将过滤后的数据保存为 NumPy 数组，存储在缓存文件中
            print(f"save cache to {cache_path}")

        else:
            cache_dict = np.load(cache_path, allow_pickle=True).item() #allow_pickle=True允许保存对象数组
            print(f"load {mode} dataset from {cache_path}")
            filtered_src = cache_dict["src"]
            filtered_trg = cache_dict["trg"]

        self.src = filtered_src
        self.trg = filtered_trg

    def __getitem__(self, index):#根据索引返回一个句子对 (源语言句子, 目标语言句子)
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)#返回数据集的大小（句子对数）


train_ds = LangPairDataset("train")
val_ds = LangPairDataset("val")
```

    save cache to wmt16/.cache/de2en_train_128.npy
    save cache to wmt16/.cache/de2en_val_128.npy



```python
!rm wmt16/.cache -r
```


```python
len(train_ds) #少了1000多个样本
```




    27465




```python
print("source: {}\ntarget: {}".format(*train_ds[-1]))
```

    source: ein älterer mann sitzt mit einem jungen mit einem wagen vor einer fa@@ ss@@ ade .
    target: an elderly man sits outside a storefront accompanied by a young boy with a cart .


### Tokenizer

这里有两种处理方式，分别对应着 encoder 和 decoder 的 word embedding 是否共享，这里实现共享的方案


```python
#载入词表，看下词表长度，词表就像英语字典,构建word2idx和idx2word
word2idx = {
    "[PAD]": 0,     # 填充 token
    "[BOS]": 1,     # begin of sentence
    "[UNK]": 2,     # 未知 token
    "[EOS]": 3,     # end of sentence
}
idx2word = {value: key for key, value in word2idx.items()}
index = len(idx2word)
threshold = 1  # 出现次数低于此的token舍弃

with open("wmt16/vocab", "r", encoding="utf8") as file:
    for line in tqdm(file.readlines()):
        token, counts = line.strip().split()
        if int(counts) >= threshold:
            word2idx[token] = index
            idx2word[index] = token
            index += 1

vocab_size = len(word2idx)
print("vocab_size: {}".format(vocab_size))
```


      0%|          | 0/9714 [00:00<?, ?it/s]


    vocab_size: 9718



```python
class Tokenizer:
    def __init__(self, word2idx, idx2word, max_length=128, pad_idx=0, bos_idx=1, eos_idx=3, unk_idx=2):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

    def encode(self, text_list, padding_first=False, add_bos=True, add_eos=True, return_mask=False):
        """如果padding_first == True，则padding加载前面，否则加载后面"""
        max_length = min(self.max_length, add_eos + add_bos + max([len(text) for text in text_list]))
        indices_list = []
        for text in text_list:
            indices = [self.word2idx.get(word, self.unk_idx) for word in text[:max_length - add_bos - add_eos]]
            if add_bos:
                indices = [self.bos_idx] + indices
            if add_eos:
                indices = indices + [self.eos_idx]
            if padding_first:
                indices = [self.pad_idx] * (max_length - len(indices)) + indices
            else:
                indices = indices + [self.pad_idx] * (max_length - len(indices))
            indices_list.append(indices)
        input_ids = torch.tensor(indices_list)
        masks = (input_ids == self.pad_idx).to(dtype=torch.int64) # 为了方便损失计算，这里的mask为0的地方需要计算，为1的地方不需要计算
        return input_ids if not return_mask else (input_ids, masks)


    def decode(self, indices_list, remove_bos=True, remove_eos=True, remove_pad=True, split=False):
        text_list = []
        for indices in indices_list:
            text = []
            for index in indices:
                word = self.idx2word.get(index, "[UNK]")
                if remove_bos and word == "[BOS]":
                    continue
                if remove_eos and word == "[EOS]":
                    break
                if remove_pad and word == "[PAD]":
                    break
                text.append(word)
            text_list.append(" ".join(text) if not split else text)
        return text_list


tokenizer = Tokenizer(word2idx=word2idx, idx2word=idx2word)

tokenizer.encode([["hello"], ["hello", "world"]], add_bos=True, add_eos=False)
raw_text = ["hello world".split(), "tokenize text datas with batch".split(), "this is a test".split()]
indices = tokenizer.encode(raw_text, padding_first=False, add_bos=True, add_eos=True)
decode_text = tokenizer.decode(indices.tolist(), remove_bos=False, remove_eos=False, remove_pad=False)
print("raw text")
for raw in raw_text:
    print(raw)
print("indices")
for index in indices:
    print(index)
print("decode text")
for decode in decode_text:
    print(decode)
```

    raw text
    ['hello', 'world']
    ['tokenize', 'text', 'datas', 'with', 'batch']
    ['this', 'is', 'a', 'test']
    indices
    tensor([   1,    2, 4517,    3,    0,    0,    0])
    tensor([   1,    2, 7167,    2,   22,    2,    3])
    tensor([   1,  425,   18,    5, 4493,    3,    0])
    decode text
    [BOS] [UNK] world [EOS] [PAD] [PAD] [PAD]
    [BOS] [UNK] text [UNK] with [UNK] [EOS]
    [BOS] this is a test [EOS] [PAD]



```python
for i,j in train_ds:
    print(len(i))
    print(len(j))
    break
```

    72
    54


### Transformer Batch Sampler

> Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens
句子按照序列长度差不多的分到一个批次。 每个训练批次包含一组句子对，其中包含大约 25000 个源标记和 25000 个目标标记


```python
class SampleInfo: #下面的info对象
    def __init__(self, i, lens):
        """
        记录文本对的序号和长度信息
        输入：
            - i (int): 文本对的序号。
            - lens (list): 文本对源语言和目标语言的长度
        """
        self.i = i
        # 加一是考虑填补在文本前后的特殊词元，lens[0]和lens[1]分别表示源语言和目标语言的长度
        self.max_len = max(lens[0], lens[1]) + 1
        self.src_len = lens[0] + 1
        self.trg_len = lens[1] + 1

# 一个批量生成器，根据词元数目的限制来控制批量的大小。它会根据传入的样本信息，在不超过设定大小的情况下，逐步构建批量。
class TokenBatchCreator:
    def __init__(self, batch_size):
        """
        参数:
        batch_size (int): 用于限制批量的大小。
        功能:
        初始化了一个空的批量列表 _batch。
        设定了初始的最大长度为 -1。
        存储了传入的 batch_size。
        """

        self._batch = []  #这个就是之前的batch_size，就是第一个batch内有多少个样本
        self.max_len = -1
        self._batch_size = batch_size # 限制批量的大小,假设是4096

    def append(self, info):
        """
        参数:
        info (SampleInfo): 文本对的信息。
        功能:
        接收一个 SampleInfo 对象，并根据其最大长度信息更新当前批量的最大长度。
        如果将新的样本加入批量后超过了批量大小限制，它会返回已有的批量并将新的样本加入新的批量。
        否则，它会更新最大长度并将样本添加到当前批量中。
        """
        # 更新当前批量的最大长度
        cur_len = info.max_len # 当前样本的长度
        max_len = max(self.max_len, cur_len) # 每来一个样本，更新当前批次的最大长度
        # 如果新的样本加入批量后超过大小限制，则将已有的批量返回，新的样本加入新的批量
        if max_len * (len(self._batch) + 1) > self._batch_size:
            self._batch, result = [], self._batch # 保存当前的batch，并返回,这里的result是之前的batch,_batch清空
            self._batch.append(info) #箱子里的第一条样本，放入
            self.max_len = cur_len #因为是当前batch的第一个样本，所以它的长度就是当前长度
            return result
        else:
            self.max_len = max_len
            self._batch.append(info) # 将样本添加到当前批量中
            return None

    @property
    def batch(self):
        return self._batch
```


```python
from torch.utils.data import BatchSampler
import numpy as np


class TransformerBatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle_batch=False,
                 clip_last_batch=False,
                 seed=0):
        """
        批量采样器
        输入:
            - dataset: 数据集
            - batch_size: 批量大小
            - shuffle_batch: 是否对生成的批量进行洗牌
            - clip_last_batch: 是否裁剪最后剩下的数据
            - seed: 随机数种子
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle_batch = shuffle_batch
        self._clip_last_batch = clip_last_batch
        self._seed = seed
        self._random = np.random
        self._random.seed(seed)

        self._sample_infos = []
        # 根据数据集中的每个样本，创建了对应的 SampleInfo 对象，包含了样本的索引和长度信息。
        for i, data in enumerate(self._dataset):
            lens = [len(data[0]), len(data[1])] #输入和输出的长度计算放到lens中
            self._sample_infos.append(SampleInfo(i, lens))

    def __iter__(self):
        """
        对数据集中的样本进行排序，排序规则是先按源语言长度排序，如果相同则按目标语言长度排序。
        使用 TokenBatchCreator 逐步组装批量数据，当满足批量大小时返回一个批量的样本信息。
        如果不裁剪最后一个批次的数据且存在剩余样本，则将这些样本组成最后一个批次。
        如果需要对批量进行洗牌，则对批次进行洗牌操作。
        通过迭代器，抛出每个批量的样本在数据集中的索引。
        """
        # 排序，如果源语言长度相同则按照目标语言的长度排列
        infos = sorted(self._sample_infos,
                       key=lambda x: (x.src_len, x.trg_len))
        # 组装批量，所有的batch都放入batch_infos
        batch_infos = []
        batch_creator = TokenBatchCreator(self._batch_size) # 批量生成器
        for info in infos:
            batch = batch_creator.append(info)
            # 存够一个batch的样本信息后，会把这个batch返回，否则返回为None
            if batch is not None:
                batch_infos.append(batch)

        # 是否抛弃最后批量的文本对
        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batch_infos.append(batch_creator.batch) # 最后一个batch

        # 打乱batch
        if self._shuffle_batch:
            self._random.shuffle(batch_infos)

        self.batch_number = len(batch_infos)
        # print(self.batch_number) #为了理解

        # 抛出一个批量的文本对在数据集中的序号
        for batch in batch_infos:
            batch_indices = [info.i for info in batch] # 批量的样本在数据集中的索引，第一个batch[0,1,.....82]，第二个batch[83,84,85,86,87]
            yield batch_indices

    def __len__(self):
        """
        返回批量的数量
        """
        if hasattr(self, "batch_number"):
            return self.batch_number
        # 计算批量的数量,没有用到下面的情况，不用看
        batch_number = (len(self._dataset) +
                        self._batch_size) // self._batch_size
        return batch_number
```


```python
sampler = TransformerBatchSampler(train_ds, batch_size=4096, shuffle_batch=True)

#为什么这里每个批量的样本对数目不一样呢？长度*batch_number>4096的时候，就会返回上一个batch，然后新的样本加入新的batch,具体要看TokenBatchCreator的44行
```


```python
for idx, batch in enumerate(sampler):
    print("第{}批量的数据中含有文本对是：{}，数量为：{}".format(idx, batch, len(batch)))
    if idx >= 3:
        break
```

    第0批量的数据中含有文本对是：[16654, 23115, 8624, 23983, 24057, 25537, 627, 3601, 6811, 25322, 517, 24268, 25656, 27038, 18056, 23425, 24868, 26952, 1091, 5938, 24083, 26179, 17015, 18312, 1002, 26094, 977, 15291, 15620, 15859, 24174, 13640, 774]，数量为：33
    第1批量的数据中含有文本对是：[19245, 24318, 26456, 21443, 15017, 24171, 12404, 591, 19399, 11306, 12442, 11707, 20146, 3980, 14219, 20359, 3751, 20670, 9803, 13542, 25447, 4032, 19577, 2495, 18092, 11924, 3003, 3613, 4457, 10856, 15712, 16946]，数量为：32
    第2批量的数据中含有文本对是：[23206, 24035, 185, 7560, 4654, 6271, 11780, 15766, 24437, 2384, 2872, 4107, 18979, 23455, 24747, 26265, 2553, 6167, 8215, 12217, 24724, 16413, 10490, 20186, 24420, 24820, 11527, 13661, 15083, 18307, 19507, 20594, 4650, 50, 6189]，数量为：35
    第3批量的数据中含有文本对是：[1910, 2491, 2901, 3346, 6241, 7451, 10576, 10941, 11386, 11490, 13271, 13716, 15608, 15631, 17403, 17614, 21062, 21407, 21410, 21810, 25086, 26958, 27449, 667, 2090, 2142, 5788, 6049, 7448, 7493, 7676, 10012, 10936, 12362, 13907, 17910, 23114, 25192, 26838, 312, 2267, 2383, 3416, 4033, 10651, 18193, 19954, 20153, 20366]，数量为：49



```python
len(sampler)
```




    530



### DataLoader


```python
def collate_fct(batch, tokenizer):
    src_words = [pair[0].split() for pair in batch]
    trg_words = [pair[1].split() for pair in batch]

    # [BOS] src [EOS] [PAD]
    encoder_inputs, encoder_inputs_mask = tokenizer.encode(
        src_words, padding_first=False, add_bos=True, add_eos=True, return_mask=True
        )

    # [BOS] trg [PAD]
    decoder_inputs = tokenizer.encode(
        trg_words, padding_first=False, add_bos=True, add_eos=False, return_mask=False,
        )

    # trg [EOS] [PAD]
    decoder_labels, decoder_labels_mask = tokenizer.encode(
        trg_words, padding_first=False, add_bos=False, add_eos=True, return_mask=True
        )

    return {
        "encoder_inputs": encoder_inputs.to(device=device),
        "encoder_inputs_mask": encoder_inputs_mask.to(device=device),
        "decoder_inputs": decoder_inputs.to(device=device),
        "decoder_labels": decoder_labels.to(device=device),
        "decoder_labels_mask": decoder_labels_mask.to(device=device),
    }

```


```python
from functools import partial # 固定collate_fct的tokenizer参数

#可以调整batch_size,来看最终的bleu
sampler = TransformerBatchSampler(train_ds, batch_size=256, shuffle_batch=True)
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
sample_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=partial(collate_fct, tokenizer=tokenizer)) #partial函数，固定collate_fct的tokenizer参数

for batch in sample_dl:
    for key, value in batch.items():
        print(key)
        print(value)
    break
```

    encoder_inputs
    tensor([[   1,   11,   25,   13, 4753,   55, 1336,  439,  900,   13,   42,  677,
                4,    3,    0],
            [   1,   11,   25,   12,    7,   17,  102,   14,   21,  304,   48,    8,
              261,    4,    3],
            [   1,    7, 4518, 2268,  153,   61,   74,   33,  233,   13,    8, 3342,
               79,    4,    3]], device='cuda:0')
    encoder_inputs_mask
    tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
    decoder_inputs
    tensor([[   1,    5,   26,   22, 4839, 1119, 1284,   15,    5,  619,  113,    4,
                0,    0,    0,    0],
            [   1,    5,   26,   19,    5,   16,   53,   15,    5,  283,    6,   75,
               20,    5,  261,    4],
            [   1,    5,   45,   19,   40,   58,   18,   63,   22,    5, 1490,   79,
                6,   10,  233,    4]], device='cuda:0')
    decoder_labels
    tensor([[   5,   26,   22, 4839, 1119, 1284,   15,    5,  619,  113,    4,    3,
                0,    0,    0,    0],
            [   5,   26,   19,    5,   16,   53,   15,    5,  283,    6,   75,   20,
                5,  261,    4,    3],
            [   5,   45,   19,   40,   58,   18,   63,   22,    5, 1490,   79,    6,
               10,  233,    4,    3]], device='cuda:0')
    decoder_labels_mask
    tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')


## 定义模型

- Transformer模型由Embedding、Transformer-Block组成
- Embedding包括：
    - WordEmbedding
    - PositionEmbedding
- Transformer-Block包括：
    - Self-Attention
    - Cross-Attention
    - MLP

### Embedding


```python

class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hyper params
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["d_model"] # 词向量维度
        self.pad_idx = config["pad_idx"]
        dropout_rate = config["dropout"]
        self.max_length = config["max_length"]

        # layers,设置padding_idx可以让pad的词向量全为0
        self.word_embedding = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=self.pad_idx
        )
        self.pos_embedding = nn.Embedding(
            self.max_length,
            self.hidden_size,
            _weight=self.get_positional_encoding(
                self.max_length, self.hidden_size
            ),# 位置编码，权重通过get_positional_encoding函数计算得到
        )
        self.pos_embedding.weight.requires_grad_(False) # 不更新位置编码的权重
        self.dropout = nn.Dropout(dropout_rate) # 随机失活层

    def get_word_embedding_weights(self):
        return self.word_embedding.weight

    # 计算位置信息
    @classmethod
    def get_positional_encoding(self, max_length, hidden_size):#max_length是最大长度，hidden_size是embedding维度相等
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, hidden_size) # 初始化位置编码
        # .unsqueeze(1) 是将这个一维张量转换为二维张量，即将其形状从 (max_length,) 变为 (max_length, 1)。这个操作在张量的维度上增加了一个维度，使其从一维变为二维，第二维的大小为 1。
        position = torch.arange(0, max_length).unsqueeze(1) # 位置信息,从0到max_length-1
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * -(torch.log(torch.Tensor([10000.0])) / hidden_size)
        )# 计算位置编码的权重,为了性能考量（是数学上的对数函数分解）
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        seq_len = input_ids.shape[1]
        assert (
            seq_len <= self.max_length
        ), f"input sequence length should no more than {self.max_length} but got {seq_len}"

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # print(position_ids)
        # embedding
        word_embeds = self.word_embedding(input_ids) # 词嵌入
        pos_embeds = self.pos_embedding(position_ids) # 位置编码
        embeds = word_embeds + pos_embeds
        embeds = self.dropout(embeds)

        return embeds


def plot_position_embedding(position_embedding):# 绘制位置编码
    plt.pcolormesh(position_embedding) # 绘制位置编码矩阵
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.colorbar() # 颜色条，-1到1的颜色范围
    plt.show()

position_embedding = TransformerEmbedding.get_positional_encoding(64, 128)
plot_position_embedding(position_embedding)

```


​    
![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_28_0.png)
​    



```python
#随机input，调用TransformerEmbedding
config={
    "vocab_size": 100,
    "d_model": 128,
    "pad_idx": 0,
    "max_length": 64,
    "dropout": 0.1,
}
input_ids = torch.randint(0, 100, (2, 50))
embeds = TransformerEmbedding(config)(input_ids)
embeds.shape
```




    torch.Size([2, 50, 128])



### Transformer Block

#### scaled-dot-product-attention


```python
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

mha = MultiHeadAttention({"num_heads": 2, "d_model": 2})
query = torch.randn(2, 3, 2) # [batch_size, seq_len, hidden_size]
query /= query.norm(dim=-1, keepdim=True) # 归一化
key_value = torch.randn(2, 4, 2)
print(f'key_value.shape {key_value.shape}')
outputs = mha(query, key_value, key_value) #最终输出shape和query的shape一样
print(outputs.hidden_states.shape)
print(outputs.attn_scores.shape)
```

    key_value.shape torch.Size([2, 4, 2])
    torch.Size([2, 3, 2])
    torch.Size([2, 2, 3, 4])



```python
# plt.subplots() 用于创建子图网格，其维度基于 outputs.attn_scores.shape[:2]。子图的行数和列数似乎由 outputs.attn_scores 的前两个维度确定。
fig, axis = plt.subplots(*outputs.attn_scores.shape[:2])
for i in range(query.shape[0]):
    for j in range(outputs.attn_scores.shape[1]):
        # axis[i, j].matshow(outputs.attn_scores[i, j].detach().numpy())：此行使用 Matplotlib 的 matshow 绘制每个 i 和 j 的注意力分数热图。detach().numpy() 将 PyTorch 张量转换为 NumPy 数组以进行可视化。
        axis[i, j].matshow(outputs.attn_scores[i, j].detach().numpy())
        for x in range(outputs.attn_scores.shape[2]):
            for y in range(outputs.attn_scores.shape[3]):
                # axis[i, j].text(y, x, f"{outputs.attn_scores[i, j, x, y]:.2f}", ha="center", va="center", color="w")：此代码在热图上叠加文本，显示 (x, y) 位置处的注意力分数。格式化部分 f"{outputs.attn_scores[i, j, x, y]:.2f}" 确保以两位小数显示注意力分数。文本以白色居中显示在 (y, x) 坐标处。
                axis[i, j].text(y, x, f"{outputs.attn_scores[i, j, x, y]:.2f}", ha="center", va="center", color="w")
fig.suptitle("multi head attention without mask")
plt.show()
```


​    
![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_33_0.png)
​    



```python
print('-'*50)
# mask
mask = torch.Tensor([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]).reshape(1, 1, 3, 4) #手工构造mask
outputs_masked = mha(query, key_value, key_value, mask)

fig, axis = plt.subplots(*outputs_masked.attn_scores.shape[:2])
for i in range(query.shape[0]):
    for j in range(outputs_masked.attn_scores.shape[1]):
        axis[i, j].matshow(outputs_masked.attn_scores[i, j].detach().numpy())
        for x in range(outputs_masked.attn_scores.shape[2]):
            for y in range(outputs_masked.attn_scores.shape[3]):
                axis[i, j].text(y, x, f"{outputs_masked.attn_scores[i, j, x, y]:.2f}", ha="center", va="center", color="w")
fig.suptitle("multi head attention with mask")
plt.show()
```

    --------------------------------------------------




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_34_1.png)
    


#### Transformer-Block


```python
# 通过使用 @dataclass 装饰器，Python 会自动为该类生成一些方法，如 __init__()、__repr__() 和 __eq__() 等，这些方法可以使类的使用更加方便。
@dataclass
class TransformerBlockOutput:
# hidden_states: Tensor：用于存储某个块产生的隐藏状态。
# self_attn_scores: Tensor：包含了自注意力机制（self-attention）所计算得到的注意力分数。
# cross_attn_scores: Optional[Tensor] = None：是一个可选字段，存储了交叉注意力（cross-attention）计算得到的注意力分数。这里的 Optional 表示这个字段可以是 Tensor 类型，也可以是 None。
    hidden_states: Tensor
    self_attn_scores: Tensor
    cross_attn_scores: Optional[Tensor] = None

class TransformerBlock(nn.Module):
    def __init__(self, config, add_cross_attention=False):
        super().__init__()
        # hyper params
        self.hidden_size = config["d_model"]
        self.num_heads = config["num_heads"]
        dropout_rate = config["dropout"]
        ffn_dim = config["dim_feedforward"]
        eps = config["layer_norm_eps"] # 层归一化的epsilon值

        # self-attention
        self.self_atten = MultiHeadAttention(config) # 多头注意力
        self.self_ln = nn.LayerNorm(self.hidden_size, eps=eps) #层归一化(层标准化)
        self.self_dropout = nn.Dropout(dropout_rate)

        # cross-attention，交叉注意力，decoder中使用,因此额外做一个判断
        if add_cross_attention:
            self.cross_atten = MultiHeadAttention(config)
            self.cross_ln = nn.LayerNorm(self.hidden_size, eps=eps)
            self.cross_dropout = nn.Dropout(dropout_rate)
        else:
            self.cross_atten = None

        # FFN,前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, self.hidden_size),
        )
        self.ffn_ln = nn.LayerNorm(self.hidden_size, eps=eps)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        attn_mask=None,
        encoder_outputs=None,
        cross_attn_mask=None,
    ):
        # self-attention,自注意力
        self_atten_output = self.self_atten(
            hidden_states, hidden_states, hidden_states, attn_mask
        )
        self_embeds = self.self_ln(
            hidden_states + self.self_dropout(self_atten_output.hidden_states)
        ) #多头注意力进行dropout，然后和原始输入进行残差连接，然后进行层归一化

        # cross-attention，交叉注意力
        if self.cross_atten is not None:
            assert encoder_outputs is not None
            cross_atten_output = self.cross_atten(
                self_embeds, encoder_outputs, encoder_outputs, cross_attn_mask
            ) #query是self_embeds，key和value都是encoder_outputs
            cross_embeds = self.cross_ln(
                self_embeds + self.cross_dropout(cross_atten_output.hidden_states)
            ) # 交叉注意力进行dropout，然后和self_embeds进行残差连接，然后进行层归一化

        # FFN
        embeds = cross_embeds if self.cross_atten is not None else self_embeds # 如果有交叉注意力，则使用交叉注意力的输出作为FFN的输入；否则，使用self_embeds作为FFN的输入
        ffn_output = self.ffn(embeds) # 前馈神经网络
        embeds = self.ffn_ln(embeds + self.ffn_dropout(ffn_output)) # 前馈神经网络进行dropout，然后和原始输入进行残差连接，然后进行层归一化

        return TransformerBlockOutput(
            hidden_states=embeds,
            self_attn_scores=self_atten_output.attn_scores,
            cross_attn_scores=cross_atten_output.attn_scores
            if self.cross_atten is not None
            else None,
        )
```

#### Encoder


```python
from typing import List

@dataclass
class TransformerEncoderOutput:
    last_hidden_states: Tensor
    attn_scores: List[Tensor]

# https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hyper params
        self.num_layers = config["num_encoder_layers"]

        # layers,仅仅是一个模块的列表，它本身没有定义前向传递（forward pass）过程。你需要在 forward 方法中明确地定义如何使用这些模块。
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_layers)]
        )

    def forward(
        self, encoder_inputs_embeds, attn_mask=None
    ) -> TransformerEncoderOutput:
        attn_scores = [] # 存储每个层的注意力分数
        embeds = encoder_inputs_embeds # 输入的嵌入向量作为第一层的输入(embedding+位置编码)
        for layer in self.layers:
            block_outputs = layer(embeds, attn_mask=attn_mask)
            embeds = block_outputs.hidden_states #上一层的输出作为下一层的输入
            # 在每个层的输出中，提取了隐藏状态 block_outputs.hidden_states，并将对应的注意力分数 block_outputs.self_attn_scores 添加到列表 attn_scores 中。
            attn_scores.append(block_outputs.self_attn_scores) # 存储每个层的注意力分数,用于画图

        return TransformerEncoderOutput(
            last_hidden_states=embeds, attn_scores=attn_scores
        )


```

#### Decoder


```python
@dataclass
class TransformerDecoderOutput:
    last_hidden_states: Tensor
    self_attn_scores: List[Tensor]
    cross_attn_scores: List[Tensor]


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hyper params
        self.num_layers = config["num_decoder_layers"]

        # layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, add_cross_attention=True)
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        decoder_inputs_embeds,
        encoder_outputs,
        attn_mask=None,
        cross_attn_mask=None,
    ) -> TransformerDecoderOutput:
        self_attn_scores = [] # 存储每个层的自注意力分数
        cross_attn_scores = [] # 存储每个层的交叉注意力分数
        embeds = decoder_inputs_embeds # 输入的嵌入向量作为第一层的输入(embedding+位置编码)
        for layer in self.layers:
            block_outputs = layer(
                embeds,
                attn_mask=attn_mask, # 自注意力的mask
                encoder_outputs=encoder_outputs,
                cross_attn_mask=cross_attn_mask, # 交叉注意力的mask
            )
            embeds = block_outputs.hidden_states # 上一层的输出作为下一层的输入
            self_attn_scores.append(block_outputs.self_attn_scores) # 存储每个层的自注意力分数
            cross_attn_scores.append(block_outputs.cross_attn_scores) # 存储每个层的交叉注意力分数

        return TransformerDecoderOutput(
            last_hidden_states=embeds,
            self_attn_scores=self_attn_scores,
            cross_attn_scores=cross_attn_scores,
        )

```

#### mask

- mask实际上大类上只有两种
    1. `padding_mask`：mask掉`pad_idx`，不计算损失
    2. `attention_mask`：mask掉`pad_idx`，不计算注意力分数
- Decoder的`attention_mask`和Encoder有一定的区别：
    - Encoder可以同时看见序列所有信息，故只mask掉`pad_idx`
    - Decoder只能看到在自身之前的序列的信息，故要额外mask掉自身之后的序列


```python
(torch.triu(torch.ones(5, 5)) == 0).transpose(-1,-2)
```




    tensor([[False,  True,  True,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False, False,  True],
            [False, False, False, False, False]])




```python

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with True.
        Unmasked positions are filled with False.
    """
    # torch.ones(sz, sz): 创建一个全为 1 的 sz × sz 的矩阵。
    # torch.triu(...): 使用 triu 函数取得矩阵的上三角部分，将主对角线以下部分置零。
    mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(-1, -2).bool()
    # mask = torch.triu(torch.ones(sz, sz))
    return mask


plt.matshow(generate_square_subsequent_mask(16))
plt.colorbar()
plt.xlabel("keys")
plt.ylabel("querys")
plt.title("1 means mask while 0 means unmask")
plt.show()
```


​    
![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_43_0.png)
​    



```python
#通过下面代码查看mask的效果
inputs_words = ["The quick brown fox jumps over the lazy dog .", "What does the fox say ?"]

inputs_ids, input_mask = tokenizer.encode([w.split() for w in inputs_words], return_mask=True)
for i in range(len(inputs_words)):
    decode_text = tokenizer.decode(inputs_ids[i: i+1].tolist(), remove_bos=False, remove_eos=False, remove_pad=False, split=True)[0]
    print(decode_text)
    self_attn_mask  = input_mask[i].reshape(1, -1).repeat_interleave(inputs_ids.shape[-1], dim=0)
    look_ahead_mask = generate_square_subsequent_mask(inputs_ids.shape[-1])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].matshow(self_attn_mask)
    axs[0].set_title("self_attn_mask")
    axs[0].set_yticks(range(len(decode_text)), decode_text, fontsize=6)
    axs[0].set_ylabel("querys")
    axs[0].set_xticks(range(len(decode_text)), decode_text, fontsize=6)
    axs[0].set_xlabel("keys")
    axs[1].matshow(look_ahead_mask)
    axs[1].set_title("look_ahead_mask")
    axs[1].set_yticks(range(len(decode_text)), decode_text, fontsize=6)
    axs[1].set_ylabel("querys")
    axs[1].set_xticks(range(len(decode_text)), decode_text, fontsize=6)
    axs[1].set_xlabel("keys")
    plt.show()
    print('-'*50)
```

    ['[BOS]', '[UNK]', '[UNK]', 'brown', '[UNK]', 'jumps', 'over', 'the', '[UNK]', 'dog', '.', '[EOS]']




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_44_1.png)
    


    --------------------------------------------------
    ['[BOS]', '[UNK]', 'does', 'the', '[UNK]', 'say', '?', '[EOS]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_44_3.png)
    


    --------------------------------------------------



```python
(torch.triu(torch.ones(5, 5)) == 0).transpose(-1, -2).bool()
```




    tensor([[False,  True,  True,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False, False,  True],
            [False, False, False, False, False]])




```python
#帮我随机两个[5, 1, 1, 4]与[1, 1, 4, 4]尺寸的张量，并求和
a = torch.randn(5, 1, 1, 4)
b = torch.randn(1, 1, 4, 4)
(a + b).shape
```




    torch.Size([5, 1, 4, 4])



#### Transformer Model


```python
@dataclass
class TransformerOutput:
    logits: Tensor
    encoder_last_hidden_states: Tensor
    encoder_attn_scores: List[Tensor] #画图
    decoder_last_hidden_states: Tensor
    decoder_self_attn_scores: List[Tensor] #画图
    decoder_cross_attn_scores: List[Tensor] #画图
    preds: Optional[Tensor] = None

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hyper params
        self.hidden_size = config["d_model"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.num_decoder_layers = config["num_decoder_layers"]
        self.pad_idx = config["pad_idx"]
        self.bos_idx = config["bos_idx"]
        self.eos_idx = config["eos_idx"]
        self.vocab_size = config["vocab_size"]
        self.dropout_rate = config["dropout"]
        self.max_length = config["max_length"]
        self.share = config["share_embedding"]

        # layers
        self.src_embedding = TransformerEmbedding(config) # 输入的嵌入层
        if self.share:#如果共享词嵌入，则使用src_embedding作为trg_embedding
            self.trg_embedding = self.src_embedding #源和目标的嵌入层相同，共享参数，节省内存
            self.linear = lambda x: torch.matmul(
                x, self.trg_embedding.get_word_embedding_weights().T
            ) # 输出层，共享参数，直接拿原有embedding矩阵的转置，节省内存
        else:
            self.trg_embedding = TransformerEmbedding(config) #decoder模块的嵌入层
            self.linear = nn.Linear(self.hidden_size, self.vocab_size) # 输出层

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        # init weights
        self._init_weights()

    def _init_weights(self):
        """使用 xavier 均匀分布来初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generate a square mask for the sequence. The masked positions are filled with True.
            Unmasked positions are filled with False.为了生成斜三角的mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(-1, -2).bool()

        return mask

    def forward(
        self, encoder_inputs, decoder_inputs, encoder_inputs_mask=None
    ) -> TransformerOutput:
        # encoder_inputs: [batch_size, src_len]
        # decoder_inputs: [batch_size, trg_len]
        # encoder_inputs_mask: [batch_size, src_len]
        if encoder_inputs_mask is None:
            encoder_inputs_mask = encoder_inputs.eq(self.pad_idx) # [batch_size, src_len]
        encoder_inputs_mask = encoder_inputs_mask.unsqueeze(1).unsqueeze(
            2
        )  # [batch_size, 1, 1, src_len],用于encoder的自注意力
        look_ahead_mask = self.generate_square_subsequent_mask(decoder_inputs.shape[1])
        look_ahead_mask = (
            look_ahead_mask.unsqueeze(0).unsqueeze(0).to(decoder_inputs.device)
        )  #[trg_len, trg_len]--> [1, 1, trg_len, trg_len],用于decoder的自注意力
        #增加decoder_inputs_mask和look_ahead_mask进行组合
        decoder_inputs_mask = decoder_inputs.eq(self.pad_idx) # [batch_size, trg_len]，和上面encoder_inputs_mask一致
        # print(decoder_inputs_mask.shape)
        decoder_inputs_mask = decoder_inputs_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, trg_len]
        # print(decoder_inputs_mask.shape)
        decoder_inputs_mask = decoder_inputs_mask + look_ahead_mask # [batch_size, 1, 1, trg_len]与[1, 1, trg_len, trg_len]相加，得到decoder的自注意力mask

        # encoding
        encoder_inputs_embeds = self.src_embedding(encoder_inputs)
        encoder_outputs = self.encoder(encoder_inputs_embeds, encoder_inputs_mask) #encoder_inputs_mask用于encoder的自注意力,广播去做计算

        # decoding
        decoder_inputs_embeds = self.trg_embedding(decoder_inputs)
        decoder_outputs = self.decoder(
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs.last_hidden_states,
            attn_mask=decoder_inputs_mask, #用于decoder的自注意力,广播去做计算
            cross_attn_mask=encoder_inputs_mask,#用于decoder的交叉注意力,广播去做计算
        )

        logits = self.linear(decoder_outputs.last_hidden_states) # [batch_size, trg_len, vocab_size]

        return TransformerOutput(
            logits=logits,
            encoder_last_hidden_states=encoder_outputs.last_hidden_states,
            encoder_attn_scores=encoder_outputs.attn_scores,
            decoder_last_hidden_states=decoder_outputs.last_hidden_states,
            decoder_self_attn_scores=decoder_outputs.self_attn_scores,
            decoder_cross_attn_scores=decoder_outputs.cross_attn_scores,
        )

    @torch.no_grad()
    def infer(self, encoder_inputs, encoder_inputs_mask=None) -> Tensor:
        # assert len(encoder_inputs.shape) == 2 and encoder_inputs.shape[0] == 1
        if encoder_inputs_mask is None:#应对多个样本同时进行推理
            encoder_inputs_mask = encoder_inputs.eq(self.pad_idx)
        encoder_inputs_mask = encoder_inputs_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len],[1,src_len]相加时，会自动广播到[batch_size,1,src_len,src_len]
        look_ahead_mask = self.generate_square_subsequent_mask(self.max_length)
        look_ahead_mask = (
            look_ahead_mask.unsqueeze(0).unsqueeze(0).to(encoder_inputs.device)
        )  # [1, 1, trg_len, trg_len]

        # encoding
        encoder_inputs_embeds = self.src_embedding(encoder_inputs)
        encoder_outputs = self.encoder(encoder_inputs_embeds) #因为只支持单样本预测，没有paddings，所以不需要mask

        # decoding,多样本推理
        decoder_inputs = torch.Tensor([self.bos_idx] * encoder_inputs.shape[0]).reshape(-1, 1).long().to(device=encoder_inputs.device)
        for cur_len in tqdm(range(1, self.max_length + 1)):
            decoder_inputs_embeds = self.trg_embedding(decoder_inputs)
            decoder_outputs = self.decoder(
                decoder_inputs_embeds=decoder_inputs_embeds,
                encoder_outputs=encoder_outputs.last_hidden_states,
                attn_mask=look_ahead_mask[:, :, :cur_len, :cur_len],#decoder的自注意力mask
            )

            logits = self.linear(decoder_outputs.last_hidden_states)
            next_token = logits.argmax(dim=-1)[:, -1:] #通过最大下标确定类别，[:, -1:]表示取最后一个结果
            decoder_inputs = torch.cat([decoder_inputs, next_token], dim=-1) #预测输出拼接到输入中
            #(decoder_inputs == self.eos_idx).sum(dim=-1)是判断样本中是否含有EOS标记
            #all是每一个都为True，才会结束
            if all((decoder_inputs == self.eos_idx).sum(dim=-1) > 0):
                break

        return TransformerOutput(
            preds=decoder_inputs[:, 1:],
            logits=logits,
            encoder_last_hidden_states=encoder_outputs.last_hidden_states,
            encoder_attn_scores=encoder_outputs.attn_scores,
            decoder_last_hidden_states=decoder_outputs.last_hidden_states,
            decoder_self_attn_scores=decoder_outputs.self_attn_scores,
            decoder_cross_attn_scores=decoder_outputs.cross_attn_scores,
        )
```

## 训练

### 损失函数


```python
class CrossEntropyWithPadding:
    def __init__(self, config):
        self.label_smoothing = config["label_smoothing"]

    def __call__(self, logits, labels, padding_mask=None):
        # logits.shape = [batch size, sequence length, num of classes]
        # labels.shape = [batch size, sequence length]
        # padding_mask.shape = [batch size, sequence length]
        bs, seq_len, nc = logits.shape
        loss = F.cross_entropy(logits.reshape(bs * seq_len, nc), labels.reshape(-1), reduce=False, label_smoothing=self.label_smoothing) #label_smoothing表示随机将一个类别的概率设置为0.1，使得模型更加关注其他类别
        if padding_mask is None:
            loss = loss.mean()
        else:
            padding_mask = 1 - padding_mask.reshape(-1) #将padding_mask reshape成一维张量，mask部分为0，非mask部分为1
            loss = torch.mul(loss, padding_mask).sum() / padding_mask.sum()

        return loss

```

### 学习率衰减


```python
x=np.arange(1, 40000)
plt.plot(x, x * (4000 ** (-1.5)))
```




    [<matplotlib.lines.Line2D at 0x7f5159ae6200>]




​    
![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_53_1.png)
​    



```python
np.sqrt(512)
```




    22.627416997969522




```python
# NoamDecayScheduler 是一个自定义或外部定义的学习率衰减调度器类。它需要接收配置 config 作为参数，可能实现了特定的学习率衰减方案
class NoamDecayScheduler:
    def __init__(self, config):
        self.d_model = config["d_model"]
        self.warmup_steps = config["warmup_steps"]

    def __call__(self, step):
        step += 1
        arg1 = step ** (-0.5) #4000步之后是arg1
        arg2 = step * (self.warmup_steps ** (-1.5))  #4000步之前是arg2

        arg3 = self.d_model ** (-0.5)

        return arg3 * np.minimum(arg1, arg2)


temp_learning_rate_schedule = NoamDecayScheduler({"d_model": 512, "warmup_steps": 4000})
#下面是学习率的设计图
plt.plot(temp_learning_rate_schedule(np.arange(0, 40000)))
plt.ylabel("Leraning rate")
plt.xlabel("Train step")
plt.show()

```


​    
![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_55_0.png)
​    


### 优化器


```python
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam

def get_optimizer(model, config):
    base_lr = 0.1
    beta1 = config["beta1"] # Adam 的 beta1
    beta2 = config["beta2"] # Adam 的 beta2
    eps = config["eps"]
    optimizer = Adam(model.parameters(), lr=base_lr, betas=(beta1, beta2), eps=eps)
    lr_scheduler = NoamDecayScheduler(config) #config是一个字典，包含了学习率衰减的参数
    # 使用 LambdaLR 调度器，它可以根据给定的函数 lr_lambda 调整学习率。这里将 lr_scheduler 作为函数传递给 LambdaLR，它包含了特定于模型或任务的学习率调度规则
    scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler)
    return optimizer, scheduler
```

### Callback


```python
from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    def __init__(self, log_dir, flush_secs=10):
        """
        Args:
            log_dir (str): dir to write log.
            flush_secs (int, optional): write to dsk each flush_secs seconds. Defaults to 10.
        """
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def draw_model(self, model, input_shape):
        self.writer.add_graph(model, input_to_model=torch.randn(input_shape))

    def add_loss_scalars(self, step, loss, val_loss):
        self.writer.add_scalars(
            main_tag="training/loss",
            tag_scalar_dict={"loss": loss, "val_loss": val_loss},
            global_step=step,
            )

    def add_acc_scalars(self, step, acc, val_acc):
        self.writer.add_scalars(
            main_tag="training/accuracy",
            tag_scalar_dict={"accuracy": acc, "val_accuracy": val_acc},
            global_step=step,
        )

    def add_lr_scalars(self, step, learning_rate):
        self.writer.add_scalars(
            main_tag="training/learning_rate",
            tag_scalar_dict={"learning_rate": learning_rate},
            global_step=step,

        )

    def __call__(self, step, **kwargs):
        # add loss
        loss = kwargs.pop("loss", None)
        val_loss = kwargs.pop("val_loss", None)
        if loss is not None and val_loss is not None:
            self.add_loss_scalars(step, loss, val_loss)
        # add acc
        acc = kwargs.pop("acc", None)
        val_acc = kwargs.pop("val_acc", None)
        if acc is not None and val_acc is not None:
            self.add_acc_scalars(step, acc, val_acc)
        # add lr
        learning_rate = kwargs.pop("lr", None)
        if learning_rate is not None:
            self.add_lr_scalars(step, learning_rate)

```


```python
class SaveCheckpointsCallback:
    def __init__(self, save_dir, save_step=5000, save_best_only=True):
        """
        Save checkpoints each save_epoch epoch.
        We save checkpoint by epoch in this implementation.
        Usually, training scripts with pytorch evaluating model and save checkpoint by step.

        Args:
            save_dir (str): dir to save checkpoint
            save_epoch (int, optional): the frequency to save checkpoint. Defaults to 1.
            save_best_only (bool, optional): If True, only save the best model or save each model at every epoch.
        """
        self.save_dir = save_dir
        self.save_step = save_step
        self.save_best_only = save_best_only
        self.best_metrics = - np.inf

        # mkdir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def __call__(self, step, state_dict, metric=None):
        if step % self.save_step > 0:
            return

        if self.save_best_only:
            assert metric is not None
            if metric >= self.best_metrics:
                # save checkpoints
                torch.save(state_dict, os.path.join(self.save_dir, "best.ckpt"))
                # update best metrics
                self.best_metrics = metric
        else:
            torch.save(state_dict, os.path.join(self.save_dir, f"{step}.ckpt"))


```


```python
class EarlyStopCallback:
    def __init__(self, patience=5, min_delta=0.01):
        """

        Args:
            patience (int, optional): Number of epochs with no improvement after which training will be stopped.. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
                change of less than min_delta, will count as no improvement. Defaults to 0.01.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = - np.inf
        self.counter = 0

    def __call__(self, metric):
        if metric >= self.best_metric + self.min_delta:
            # update best metric
            self.best_metric = metric
            # reset counter
            self.counter = 0
        else:
            self.counter += 1

    @property
    def early_stop(self):
        return self.counter >= self.patience

```

### training & valuating


```python
@torch.no_grad()
def evaluating(model, dataloader, loss_fct):
    loss_list = []
    for batch in dataloader:
        encoder_inputs = batch["encoder_inputs"]
        encoder_inputs_mask = batch["encoder_inputs_mask"]
        decoder_inputs = batch["decoder_inputs"]
        decoder_labels = batch["decoder_labels"]
        decoder_labels_mask = batch["decoder_labels_mask"]

        # 前向计算
        outputs = model(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            encoder_inputs_mask=encoder_inputs_mask
            )
        logits = outputs.logits
        loss = loss_fct(logits, decoder_labels, padding_mask=decoder_labels_mask)         # 验证集损失
        loss_list.append(loss.cpu().item())

    return np.mean(loss_list)

```


```python
# 训练
def training(
    model,
    train_loader,
    val_loader,
    epoch,
    loss_fct,
    optimizer,
    scheduler=None,
    tensorboard_callback=None,
    save_ckpt_callback=None,
    early_stop_callback=None,
    eval_step=500,
    ):
    record_dict = {
        "train": [],
        "val": []
    }

    global_step = 1
    model.train()
    with tqdm(total=epoch * len(train_loader)) as pbar:
        for epoch_id in range(epoch):
            # training
            for batch in train_loader:
                encoder_inputs = batch["encoder_inputs"]
                encoder_inputs_mask = batch["encoder_inputs_mask"]
                decoder_inputs = batch["decoder_inputs"]
                decoder_labels = batch["decoder_labels"]
                decoder_labels_mask = batch["decoder_labels_mask"]
                # 梯度清空
                optimizer.zero_grad()

                # 前向计算
                outputs = model(
                    encoder_inputs=encoder_inputs,
                    decoder_inputs=decoder_inputs,
                    encoder_inputs_mask=encoder_inputs_mask
                    )
                logits = outputs.logits
                loss = loss_fct(logits, decoder_labels, padding_mask=decoder_labels_mask)

                # 梯度回传
                loss.backward()

                # 调整优化器，包括学习率的变动等
                optimizer.step()
                if scheduler is not None:
                    scheduler.step() # 更新学习率

                loss = loss.cpu().item()
                # record
                record_dict["train"].append({
                    "loss": loss, "step": global_step
                })

                # evaluating
                if global_step % eval_step == 0:
                    model.eval()
                    val_loss = evaluating(model, val_loader, loss_fct)
                    record_dict["val"].append({
                        "loss": val_loss, "step": global_step
                    })
                    model.train()

                    # 1. 使用 tensorboard 可视化
                    cur_lr = optimizer.param_groups[0]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
                    if tensorboard_callback is not None:
                        tensorboard_callback(
                            global_step,
                            loss=loss, val_loss=val_loss,
                            lr=cur_lr,
                            )

                    # 2. 保存模型权重 save model checkpoint
                    if save_ckpt_callback is not None:
                        save_ckpt_callback(global_step, model.state_dict(), metric=-val_loss)

                    # 3. 早停 Early Stop
                    if early_stop_callback is not None:
                        early_stop_callback(-val_loss)
                        if early_stop_callback.early_stop:
                            print(f"Early stop at epoch {epoch_id} / global_step {global_step}")
                            return record_dict

                # udate step
                global_step += 1
                pbar.update(1)
            pbar.set_postfix({"epoch": epoch_id, "loss": loss, "val_loss": val_loss})

    return record_dict

```


```python
#模型的超参
config = {
    "bos_idx": 1,
    "eos_idx": 3,
    "pad_idx": 0,
    "vocab_size": len(word2idx),
    "max_length": 128,
    "d_model": 512,
    "dim_feedforward": 2048, # FFN 的隐藏层大小
    "dropout": 0.1,
    "layer_norm_eps": 1e-6, # 层归一化的 epsilon, 防止除零错误
    "num_heads": 8,
    "num_decoder_layers": 6,
    "num_encoder_layers": 6,
    "label_smoothing": 0.1,
    "beta1": 0.9, # Adam 的 beta1
    "beta2": 0.98,
    "eps": 1e-9,
    "warmup_steps": 4_000,
    "share_embedding": False, # 是否共享词向量
    }


def get_dl(dataset, batch_size, shuffle=True):
    sampler = TransformerBatchSampler(dataset, batch_size=batch_size, shuffle_batch=shuffle)
    sample_dl = DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(collate_fct, tokenizer=tokenizer))
    return sample_dl

# dataset
train_ds = LangPairDataset("train", max_length=config["max_length"])
val_ds = LangPairDataset("val", max_length=config["max_length"])
# tokenizer
tokenizer = Tokenizer(word2idx=word2idx, idx2word=idx2word, max_length=config["max_length"])
batch_size = 2048
# dataloader
train_dl = get_dl(train_ds, batch_size=batch_size, shuffle=True)
val_dl = get_dl(val_ds, batch_size=batch_size, shuffle=False)
```

    save cache to wmt16/.cache/de2en_train_128.npy
    save cache to wmt16/.cache/de2en_val_128.npy



```python
#计算模型参数量
model = TransformerModel(config)
print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

    模型参数量: 59038198



```python
config
```




    {'bos_idx': 1,
     'eos_idx': 3,
     'pad_idx': 0,
     'vocab_size': 9718,
     'max_length': 128,
     'd_model': 512,
     'dim_feedforward': 2048,
     'dropout': 0.1,
     'layer_norm_eps': 1e-06,
     'num_heads': 8,
     'num_decoder_layers': 6,
     'num_encoder_layers': 6,
     'label_smoothing': 0.1,
     'beta1': 0.9,
     'beta2': 0.98,
     'eps': 1e-09,
     'warmup_steps': 4000,
     'share_embedding': False}




```python
epoch = 100

# model
model = TransformerModel(config)
# 1. 定义损失函数 采用交叉熵损失
loss_fct = CrossEntropyWithPadding(config)
# 2. 定义优化器 采用 adam
# Optimizers specified in the torch.optim package
optimizer, scheduler = get_optimizer(model, config)

# 1. tensorboard 可视化
if not os.path.exists("runs"):
    os.mkdir("runs")
exp_name = "translate-transformer-{}".format("share" if config["share_embedding"] else "not-share")
tensorboard_callback = TensorBoardCallback(f"runs/{exp_name}")
# tensorboard_callback.draw_model(model, [1, MAX_LENGTH])
# 2. save best
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
save_ckpt_callback = SaveCheckpointsCallback(
    f"checkpoints/{exp_name}", save_step=500, save_best_only=True)
# 3. early stop
early_stop_callback = EarlyStopCallback(patience=10,min_delta=0.001)

model = model.to(device)




# We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).
```


```python
record = training(
    model,
    train_dl,
    val_dl,
    epoch,
    loss_fct,
    optimizer,
    scheduler,
    tensorboard_callback=tensorboard_callback,
    save_ckpt_callback=save_ckpt_callback,
    early_stop_callback=early_stop_callback,
    eval_step=500
    )
```


      0%|          | 0/1400 [00:00<?, ?it/s]


    /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages/torch/nn/_reduction.py:51: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
      warnings.warn(warning.format(ret))


    Early stop at epoch 30 / global_step 32000



```python
record
```




    {'train': [{'loss': 9.188334465026855, 'step': 1},
      {'loss': 9.163097381591797, 'step': 2},
      {'loss': 9.156265258789062, 'step': 3},
      {'loss': 9.163603782653809, 'step': 4},
      {'loss': 9.161243438720703, 'step': 5},
      {'loss': 9.145940780639648, 'step': 6},
      {'loss': 9.183747291564941, 'step': 7},
      {'loss': 9.169991493225098, 'step': 8},
      {'loss': 9.154008865356445, 'step': 9},
      {'loss': 9.139060974121094, 'step': 10},
      {'loss': 9.174156188964844, 'step': 11},
      {'loss': 9.163369178771973, 'step': 12},
      {'loss': 9.165631294250488, 'step': 13},
      {'loss': 9.158583641052246, 'step': 14},
      {'loss': 9.158117294311523, 'step': 15},
      {'loss': 9.156033515930176, 'step': 16},
      {'loss': 9.138397216796875, 'step': 17},
      {'loss': 9.149402618408203, 'step': 18},
      {'loss': 9.13708782196045, 'step': 19},
      {'loss': 9.148542404174805, 'step': 20},
      {'loss': 9.138191223144531, 'step': 21},
      {'loss': 9.124444007873535, 'step': 22},
      {'loss': 9.135567665100098, 'step': 23},
      {'loss': 9.100475311279297, 'step': 24},
      {'loss': 9.117512702941895, 'step': 25},
      {'loss': 9.134199142456055, 'step': 26},
      {'loss': 9.10678768157959, 'step': 27},
      {'loss': 9.10887336730957, 'step': 28},
      {'loss': 9.109654426574707, 'step': 29},
      {'loss': 9.08731746673584, 'step': 30},
      {'loss': 9.073083877563477, 'step': 31},
      {'loss': 9.07047176361084, 'step': 32},
      {'loss': 9.11690616607666, 'step': 33},
      {'loss': 9.115464210510254, 'step': 34},
      {'loss': 9.058672904968262, 'step': 35},
      {'loss': 9.07282543182373, 'step': 36},
      {'loss': 9.061594009399414, 'step': 37},
      {'loss': 9.053346633911133, 'step': 38},
      {'loss': 9.09089469909668, 'step': 39},
      {'loss': 9.037969589233398, 'step': 40},
      {'loss': 9.018332481384277, 'step': 41},
      {'loss': 9.075372695922852, 'step': 42},
      {'loss': 9.03366470336914, 'step': 43},
      {'loss': 9.0562744140625, 'step': 44},
      {'loss': 9.033679962158203, 'step': 45},
      {'loss': 9.057821273803711, 'step': 46},
      {'loss': 8.998479843139648, 'step': 47},
      {'loss': 8.970792770385742, 'step': 48},
      {'loss': 8.926719665527344, 'step': 49},
      {'loss': 8.97944450378418, 'step': 50},
      {'loss': 9.002727508544922, 'step': 51},
      {'loss': 9.012484550476074, 'step': 52},
      {'loss': 8.959132194519043, 'step': 53},
      {'loss': 8.942873001098633, 'step': 54},
      {'loss': 8.96621322631836, 'step': 55},
      {'loss': 8.991101264953613, 'step': 56},
      {'loss': 9.031423568725586, 'step': 57},
      {'loss': 8.972515106201172, 'step': 58},
      {'loss': 8.937674522399902, 'step': 59},
      {'loss': 8.978586196899414, 'step': 60},
      {'loss': 8.939094543457031, 'step': 61},
      {'loss': 8.9097318649292, 'step': 62},
      {'loss': 8.92475700378418, 'step': 63},
      {'loss': 8.88808822631836, 'step': 64},
      {'loss': 8.931924819946289, 'step': 65},
      {'loss': 8.92566204071045, 'step': 66},
      {'loss': 8.853536605834961, 'step': 67},
      {'loss': 8.939233779907227, 'step': 68},
      {'loss': 8.827993392944336, 'step': 69},
      {'loss': 8.890874862670898, 'step': 70},
      {'loss': 8.807721138000488, 'step': 71},
      {'loss': 8.88418197631836, 'step': 72},
      {'loss': 8.808655738830566, 'step': 73},
      {'loss': 8.828412055969238, 'step': 74},
      {'loss': 8.882791519165039, 'step': 75},
      {'loss': 8.863835334777832, 'step': 76},
      {'loss': 8.858354568481445, 'step': 77},
      {'loss': 8.711225509643555, 'step': 78},
      {'loss': 8.726024627685547, 'step': 79},
      {'loss': 8.74549674987793, 'step': 80},
      {'loss': 8.860406875610352, 'step': 81},
      {'loss': 8.829627990722656, 'step': 82},
      {'loss': 8.754029273986816, 'step': 83},
      {'loss': 8.663451194763184, 'step': 84},
      {'loss': 8.767935752868652, 'step': 85},
      {'loss': 8.628543853759766, 'step': 86},
      {'loss': 8.707825660705566, 'step': 87},
      {'loss': 8.632133483886719, 'step': 88},
      {'loss': 8.732610702514648, 'step': 89},
      {'loss': 8.721258163452148, 'step': 90},
      {'loss': 8.704803466796875, 'step': 91},
      {'loss': 8.716176986694336, 'step': 92},
      {'loss': 8.599668502807617, 'step': 93},
      {'loss': 8.68143367767334, 'step': 94},
      {'loss': 8.687503814697266, 'step': 95},
      {'loss': 8.681896209716797, 'step': 96},
      {'loss': 8.6771879196167, 'step': 97},
      {'loss': 8.679055213928223, 'step': 98},
      {'loss': 8.647006034851074, 'step': 99},
      {'loss': 8.772719383239746, 'step': 100},
      {'loss': 8.493697166442871, 'step': 101},
      {'loss': 8.7255220413208, 'step': 102},
      {'loss': 8.633460998535156, 'step': 103},
      {'loss': 8.600658416748047, 'step': 104},
      {'loss': 8.597679138183594, 'step': 105},
      {'loss': 8.634135246276855, 'step': 106},
      {'loss': 8.533317565917969, 'step': 107},
      {'loss': 8.680813789367676, 'step': 108},
      {'loss': 8.666993141174316, 'step': 109},
      {'loss': 8.570330619812012, 'step': 110},
      {'loss': 8.613409042358398, 'step': 111},
      {'loss': 8.753705024719238, 'step': 112},
      {'loss': 8.502416610717773, 'step': 113},
      {'loss': 8.591278076171875, 'step': 114},
      {'loss': 8.520708084106445, 'step': 115},
      {'loss': 8.543121337890625, 'step': 116},
      {'loss': 8.635903358459473, 'step': 117},
      {'loss': 8.441516876220703, 'step': 118},
      {'loss': 8.481791496276855, 'step': 119},
      {'loss': 8.459712982177734, 'step': 120},
      {'loss': 8.611433029174805, 'step': 121},
      {'loss': 8.70238208770752, 'step': 122},
      {'loss': 8.657255172729492, 'step': 123},
      {'loss': 8.474924087524414, 'step': 124},
      {'loss': 8.59689998626709, 'step': 125},
      {'loss': 8.463504791259766, 'step': 126},
      {'loss': 8.368735313415527, 'step': 127},
      {'loss': 8.44383716583252, 'step': 128},
      {'loss': 8.543118476867676, 'step': 129},
      {'loss': 8.497199058532715, 'step': 130},
      {'loss': 8.554336547851562, 'step': 131},
      {'loss': 8.553304672241211, 'step': 132},
      {'loss': 8.545211791992188, 'step': 133},
      {'loss': 8.600922584533691, 'step': 134},
      {'loss': 8.399214744567871, 'step': 135},
      {'loss': 8.527531623840332, 'step': 136},
      {'loss': 8.439294815063477, 'step': 137},
      {'loss': 8.38947868347168, 'step': 138},
      {'loss': 8.480242729187012, 'step': 139},
      {'loss': 8.485621452331543, 'step': 140},
      {'loss': 8.603104591369629, 'step': 141},
      {'loss': 8.459439277648926, 'step': 142},
      {'loss': 8.327475547790527, 'step': 143},
      {'loss': 8.423924446105957, 'step': 144},
      {'loss': 8.584364891052246, 'step': 145},
      {'loss': 8.459715843200684, 'step': 146},
      {'loss': 8.369366645812988, 'step': 147},
      {'loss': 8.33295726776123, 'step': 148},
      {'loss': 8.35740852355957, 'step': 149},
      {'loss': 8.393327713012695, 'step': 150},
      {'loss': 8.26783561706543, 'step': 151},
      {'loss': 8.58085823059082, 'step': 152},
      {'loss': 8.377471923828125, 'step': 153},
      {'loss': 8.328495979309082, 'step': 154},
      {'loss': 8.474672317504883, 'step': 155},
      {'loss': 8.405061721801758, 'step': 156},
      {'loss': 8.443942070007324, 'step': 157},
      {'loss': 8.492175102233887, 'step': 158},
      {'loss': 8.443719863891602, 'step': 159},
      {'loss': 8.288125991821289, 'step': 160},
      {'loss': 8.298142433166504, 'step': 161},
      {'loss': 8.429221153259277, 'step': 162},
      {'loss': 8.34277057647705, 'step': 163},
      {'loss': 8.423338890075684, 'step': 164},
      {'loss': 8.403039932250977, 'step': 165},
      {'loss': 8.345120429992676, 'step': 166},
      {'loss': 8.300199508666992, 'step': 167},
      {'loss': 8.16256046295166, 'step': 168},
      {'loss': 8.217596054077148, 'step': 169},
      {'loss': 8.401046752929688, 'step': 170},
      {'loss': 8.20324993133545, 'step': 171},
      {'loss': 8.331937789916992, 'step': 172},
      {'loss': 8.28752326965332, 'step': 173},
      {'loss': 8.157916069030762, 'step': 174},
      {'loss': 8.281954765319824, 'step': 175},
      {'loss': 8.358172416687012, 'step': 176},
      {'loss': 8.241463661193848, 'step': 177},
      {'loss': 8.26816177368164, 'step': 178},
      {'loss': 8.438421249389648, 'step': 179},
      {'loss': 8.170012474060059, 'step': 180},
      {'loss': 8.304651260375977, 'step': 181},
      {'loss': 8.37601089477539, 'step': 182},
      {'loss': 8.157581329345703, 'step': 183},
      {'loss': 8.290129661560059, 'step': 184},
      {'loss': 8.2886381149292, 'step': 185},
      {'loss': 8.242094039916992, 'step': 186},
      {'loss': 8.119073867797852, 'step': 187},
      {'loss': 8.32127857208252, 'step': 188},
      {'loss': 8.49549674987793, 'step': 189},
      {'loss': 8.269658088684082, 'step': 190},
      {'loss': 8.086065292358398, 'step': 191},
      {'loss': 8.129916191101074, 'step': 192},
      {'loss': 8.201827049255371, 'step': 193},
      {'loss': 8.252378463745117, 'step': 194},
      {'loss': 8.342327117919922, 'step': 195},
      {'loss': 8.317118644714355, 'step': 196},
      {'loss': 8.423348426818848, 'step': 197},
      {'loss': 8.210434913635254, 'step': 198},
      {'loss': 8.066394805908203, 'step': 199},
      {'loss': 8.083707809448242, 'step': 200},
      {'loss': 8.306394577026367, 'step': 201},
      {'loss': 8.184601783752441, 'step': 202},
      {'loss': 8.210204124450684, 'step': 203},
      {'loss': 8.2587890625, 'step': 204},
      {'loss': 8.27472972869873, 'step': 205},
      {'loss': 8.385921478271484, 'step': 206},
      {'loss': 8.358994483947754, 'step': 207},
      {'loss': 8.14958381652832, 'step': 208},
      {'loss': 8.192190170288086, 'step': 209},
      {'loss': 8.267223358154297, 'step': 210},
      {'loss': 8.102173805236816, 'step': 211},
      {'loss': 8.153284072875977, 'step': 212},
      {'loss': 8.169922828674316, 'step': 213},
      {'loss': 8.015722274780273, 'step': 214},
      {'loss': 8.190160751342773, 'step': 215},
      {'loss': 8.251121520996094, 'step': 216},
      {'loss': 8.290104866027832, 'step': 217},
      {'loss': 8.113165855407715, 'step': 218},
      {'loss': 8.261683464050293, 'step': 219},
      {'loss': 8.193206787109375, 'step': 220},
      {'loss': 8.231189727783203, 'step': 221},
      {'loss': 8.300610542297363, 'step': 222},
      {'loss': 8.057364463806152, 'step': 223},
      {'loss': 8.152673721313477, 'step': 224},
      {'loss': 8.335652351379395, 'step': 225},
      {'loss': 8.240171432495117, 'step': 226},
      {'loss': 7.985518932342529, 'step': 227},
      {'loss': 8.24262523651123, 'step': 228},
      {'loss': 8.147432327270508, 'step': 229},
      {'loss': 8.166889190673828, 'step': 230},
      {'loss': 8.296622276306152, 'step': 231},
      {'loss': 8.216814041137695, 'step': 232},
      {'loss': 8.076658248901367, 'step': 233},
      {'loss': 8.01921558380127, 'step': 234},
      {'loss': 8.355704307556152, 'step': 235},
      {'loss': 8.110118865966797, 'step': 236},
      {'loss': 8.085168838500977, 'step': 237},
      {'loss': 8.044975280761719, 'step': 238},
      {'loss': 8.1134672164917, 'step': 239},
      {'loss': 7.925731658935547, 'step': 240},
      {'loss': 8.157630920410156, 'step': 241},
      {'loss': 7.897795677185059, 'step': 242},
      {'loss': 7.880152702331543, 'step': 243},
      {'loss': 8.043815612792969, 'step': 244},
      {'loss': 7.867195129394531, 'step': 245},
      {'loss': 8.062604904174805, 'step': 246},
      {'loss': 8.004666328430176, 'step': 247},
      {'loss': 8.205350875854492, 'step': 248},
      {'loss': 8.25848388671875, 'step': 249},
      {'loss': 8.076305389404297, 'step': 250},
      {'loss': 8.201144218444824, 'step': 251},
      {'loss': 7.983108997344971, 'step': 252},
      {'loss': 8.103622436523438, 'step': 253},
      {'loss': 8.263126373291016, 'step': 254},
      {'loss': 8.282876014709473, 'step': 255},
      {'loss': 8.013093948364258, 'step': 256},
      {'loss': 8.034055709838867, 'step': 257},
      {'loss': 8.072541236877441, 'step': 258},
      {'loss': 8.004026412963867, 'step': 259},
      {'loss': 8.159942626953125, 'step': 260},
      {'loss': 8.016136169433594, 'step': 261},
      {'loss': 8.060018539428711, 'step': 262},
      {'loss': 7.714187145233154, 'step': 263},
      {'loss': 8.221376419067383, 'step': 264},
      {'loss': 8.041244506835938, 'step': 265},
      {'loss': 7.9903388023376465, 'step': 266},
      {'loss': 8.095247268676758, 'step': 267},
      {'loss': 8.059784889221191, 'step': 268},
      {'loss': 8.213482856750488, 'step': 269},
      {'loss': 8.116474151611328, 'step': 270},
      {'loss': 7.847304344177246, 'step': 271},
      {'loss': 7.800411701202393, 'step': 272},
      {'loss': 7.926199913024902, 'step': 273},
      {'loss': 7.923544883728027, 'step': 274},
      {'loss': 8.163156509399414, 'step': 275},
      {'loss': 8.054810523986816, 'step': 276},
      {'loss': 8.183835983276367, 'step': 277},
      {'loss': 8.180681228637695, 'step': 278},
      {'loss': 8.011621475219727, 'step': 279},
      {'loss': 8.115416526794434, 'step': 280},
      {'loss': 7.722769737243652, 'step': 281},
      {'loss': 7.81413459777832, 'step': 282},
      {'loss': 7.884825706481934, 'step': 283},
      {'loss': 8.112889289855957, 'step': 284},
      {'loss': 7.869754314422607, 'step': 285},
      {'loss': 8.05416488647461, 'step': 286},
      {'loss': 8.024618148803711, 'step': 287},
      {'loss': 7.518795967102051, 'step': 288},
      {'loss': 8.069442749023438, 'step': 289},
      {'loss': 8.072574615478516, 'step': 290},
      {'loss': 7.913673400878906, 'step': 291},
      {'loss': 8.133858680725098, 'step': 292},
      {'loss': 7.932168483734131, 'step': 293},
      {'loss': 7.703782558441162, 'step': 294},
      {'loss': 7.714348793029785, 'step': 295},
      {'loss': 8.13187313079834, 'step': 296},
      {'loss': 7.7223286628723145, 'step': 297},
      {'loss': 8.224059104919434, 'step': 298},
      {'loss': 8.209806442260742, 'step': 299},
      {'loss': 7.962831020355225, 'step': 300},
      {'loss': 7.812789440155029, 'step': 301},
      {'loss': 7.644270420074463, 'step': 302},
      {'loss': 7.970966815948486, 'step': 303},
      {'loss': 7.870840072631836, 'step': 304},
      {'loss': 7.966115951538086, 'step': 305},
      {'loss': 8.027691841125488, 'step': 306},
      {'loss': 8.083934783935547, 'step': 307},
      {'loss': 8.058582305908203, 'step': 308},
      {'loss': 7.893897533416748, 'step': 309},
      {'loss': 7.96106481552124, 'step': 310},
      {'loss': 7.70029878616333, 'step': 311},
      {'loss': 7.899839878082275, 'step': 312},
      {'loss': 8.068272590637207, 'step': 313},
      {'loss': 7.792897701263428, 'step': 314},
      {'loss': 7.879688262939453, 'step': 315},
      {'loss': 7.986254692077637, 'step': 316},
      {'loss': 7.907270908355713, 'step': 317},
      {'loss': 7.887722969055176, 'step': 318},
      {'loss': 7.8378167152404785, 'step': 319},
      {'loss': 7.784835338592529, 'step': 320},
      {'loss': 7.98853063583374, 'step': 321},
      {'loss': 7.986330032348633, 'step': 322},
      {'loss': 7.840301990509033, 'step': 323},
      {'loss': 7.8554534912109375, 'step': 324},
      {'loss': 7.918119430541992, 'step': 325},
      {'loss': 7.964860916137695, 'step': 326},
      {'loss': 7.661999225616455, 'step': 327},
      {'loss': 7.928548812866211, 'step': 328},
      {'loss': 7.6778082847595215, 'step': 329},
      {'loss': 7.656693458557129, 'step': 330},
      {'loss': 7.693058967590332, 'step': 331},
      {'loss': 7.900586128234863, 'step': 332},
      {'loss': 7.664775371551514, 'step': 333},
      {'loss': 7.718320369720459, 'step': 334},
      {'loss': 7.928235054016113, 'step': 335},
      {'loss': 7.975431442260742, 'step': 336},
      {'loss': 7.789527893066406, 'step': 337},
      {'loss': 7.803587436676025, 'step': 338},
      {'loss': 7.715322494506836, 'step': 339},
      {'loss': 7.853007793426514, 'step': 340},
      {'loss': 7.7774739265441895, 'step': 341},
      {'loss': 7.611175060272217, 'step': 342},
      {'loss': 7.410451412200928, 'step': 343},
      {'loss': 7.764491558074951, 'step': 344},
      {'loss': 7.562377452850342, 'step': 345},
      {'loss': 7.922785758972168, 'step': 346},
      {'loss': 7.849442481994629, 'step': 347},
      {'loss': 7.800525665283203, 'step': 348},
      {'loss': 7.759942054748535, 'step': 349},
      {'loss': 7.755109786987305, 'step': 350},
      {'loss': 7.629721164703369, 'step': 351},
      {'loss': 7.711826801300049, 'step': 352},
      {'loss': 7.819555759429932, 'step': 353},
      {'loss': 7.91204833984375, 'step': 354},
      {'loss': 7.827396392822266, 'step': 355},
      {'loss': 7.9114508628845215, 'step': 356},
      {'loss': 7.587447166442871, 'step': 357},
      {'loss': 7.398324012756348, 'step': 358},
      {'loss': 7.753441333770752, 'step': 359},
      {'loss': 7.896338939666748, 'step': 360},
      {'loss': 7.645166397094727, 'step': 361},
      {'loss': 7.619353771209717, 'step': 362},
      {'loss': 7.510433197021484, 'step': 363},
      {'loss': 7.706107139587402, 'step': 364},
      {'loss': 7.36716890335083, 'step': 365},
      {'loss': 7.314425468444824, 'step': 366},
      {'loss': 7.39312219619751, 'step': 367},
      {'loss': 7.473427772521973, 'step': 368},
      {'loss': 7.750062465667725, 'step': 369},
      {'loss': 7.302928924560547, 'step': 370},
      {'loss': 7.703786849975586, 'step': 371},
      {'loss': 7.62907075881958, 'step': 372},
      {'loss': 7.415689468383789, 'step': 373},
      {'loss': 7.719257831573486, 'step': 374},
      {'loss': 7.8298444747924805, 'step': 375},
      {'loss': 7.702939033508301, 'step': 376},
      {'loss': 7.73544979095459, 'step': 377},
      {'loss': 7.680583953857422, 'step': 378},
      {'loss': 7.8138885498046875, 'step': 379},
      {'loss': 7.5626091957092285, 'step': 380},
      {'loss': 7.290988922119141, 'step': 381},
      {'loss': 7.873232364654541, 'step': 382},
      {'loss': 7.445882320404053, 'step': 383},
      {'loss': 7.356894016265869, 'step': 384},
      {'loss': 7.632788181304932, 'step': 385},
      {'loss': 7.565016746520996, 'step': 386},
      {'loss': 7.831039905548096, 'step': 387},
      {'loss': 7.624150276184082, 'step': 388},
      {'loss': 7.465118885040283, 'step': 389},
      {'loss': 7.783608913421631, 'step': 390},
      {'loss': 7.4623565673828125, 'step': 391},
      {'loss': 7.397500038146973, 'step': 392},
      {'loss': 7.341700077056885, 'step': 393},
      {'loss': 7.313669204711914, 'step': 394},
      {'loss': 7.402578830718994, 'step': 395},
      {'loss': 7.491714000701904, 'step': 396},
      {'loss': 7.099094390869141, 'step': 397},
      {'loss': 7.391056060791016, 'step': 398},
      {'loss': 7.212550163269043, 'step': 399},
      {'loss': 7.654384136199951, 'step': 400},
      {'loss': 7.346584320068359, 'step': 401},
      {'loss': 7.4175028800964355, 'step': 402},
      {'loss': 7.414111614227295, 'step': 403},
      {'loss': 7.504945278167725, 'step': 404},
      {'loss': 7.7660980224609375, 'step': 405},
      {'loss': 7.110056400299072, 'step': 406},
      {'loss': 7.525433540344238, 'step': 407},
      {'loss': 7.6162872314453125, 'step': 408},
      {'loss': 7.505929946899414, 'step': 409},
      {'loss': 7.7176127433776855, 'step': 410},
      {'loss': 7.644706726074219, 'step': 411},
      {'loss': 7.3292436599731445, 'step': 412},
      {'loss': 7.33786153793335, 'step': 413},
      {'loss': 7.222848892211914, 'step': 414},
      {'loss': 7.9047746658325195, 'step': 415},
      {'loss': 7.393548965454102, 'step': 416},
      {'loss': 7.386056900024414, 'step': 417},
      {'loss': 7.423888206481934, 'step': 418},
      {'loss': 7.273349761962891, 'step': 419},
      {'loss': 7.321216106414795, 'step': 420},
      {'loss': 7.619138717651367, 'step': 421},
      {'loss': 7.168022632598877, 'step': 422},
      {'loss': 7.147271156311035, 'step': 423},
      {'loss': 7.409670829772949, 'step': 424},
      {'loss': 7.621241569519043, 'step': 425},
      {'loss': 7.236505031585693, 'step': 426},
      {'loss': 7.373208045959473, 'step': 427},
      {'loss': 7.689337253570557, 'step': 428},
      {'loss': 7.181066989898682, 'step': 429},
      {'loss': 7.275871276855469, 'step': 430},
      {'loss': 7.607172012329102, 'step': 431},
      {'loss': 7.639070987701416, 'step': 432},
      {'loss': 7.290582656860352, 'step': 433},
      {'loss': 7.040691375732422, 'step': 434},
      {'loss': 7.452157020568848, 'step': 435},
      {'loss': 7.152422904968262, 'step': 436},
      {'loss': 7.1166462898254395, 'step': 437},
      {'loss': 7.321671485900879, 'step': 438},
      {'loss': 7.430654048919678, 'step': 439},
      {'loss': 7.277205467224121, 'step': 440},
      {'loss': 7.542855739593506, 'step': 441},
      {'loss': 7.5221147537231445, 'step': 442},
      {'loss': 7.45583963394165, 'step': 443},
      {'loss': 7.422186851501465, 'step': 444},
      {'loss': 7.583787441253662, 'step': 445},
      {'loss': 7.292235374450684, 'step': 446},
      {'loss': 7.604977607727051, 'step': 447},
      {'loss': 7.0268874168396, 'step': 448},
      {'loss': 7.391992568969727, 'step': 449},
      {'loss': 7.552767753601074, 'step': 450},
      {'loss': 7.432605743408203, 'step': 451},
      {'loss': 7.126698970794678, 'step': 452},
      {'loss': 7.230589866638184, 'step': 453},
      {'loss': 7.486452579498291, 'step': 454},
      {'loss': 7.189754962921143, 'step': 455},
      {'loss': 7.091473579406738, 'step': 456},
      {'loss': 7.17783784866333, 'step': 457},
      {'loss': 7.565439701080322, 'step': 458},
      {'loss': 7.04803991317749, 'step': 459},
      {'loss': 7.325812816619873, 'step': 460},
      {'loss': 7.646679401397705, 'step': 461},
      {'loss': 7.416247367858887, 'step': 462},
      {'loss': 7.063977241516113, 'step': 463},
      {'loss': 7.189974308013916, 'step': 464},
      {'loss': 7.100983619689941, 'step': 465},
      {'loss': 7.458994388580322, 'step': 466},
      {'loss': 7.383293151855469, 'step': 467},
      {'loss': 7.066051483154297, 'step': 468},
      {'loss': 7.281315803527832, 'step': 469},
      {'loss': 7.345914363861084, 'step': 470},
      {'loss': 7.329463005065918, 'step': 471},
      {'loss': 7.36545991897583, 'step': 472},
      {'loss': 7.1801652908325195, 'step': 473},
      {'loss': 7.423101425170898, 'step': 474},
      {'loss': 6.97535514831543, 'step': 475},
      {'loss': 7.125184059143066, 'step': 476},
      {'loss': 7.02192497253418, 'step': 477},
      {'loss': 7.099111557006836, 'step': 478},
      {'loss': 7.009823322296143, 'step': 479},
      {'loss': 6.97028112411499, 'step': 480},
      {'loss': 7.221836090087891, 'step': 481},
      {'loss': 7.409386157989502, 'step': 482},
      {'loss': 7.3144683837890625, 'step': 483},
      {'loss': 7.463541030883789, 'step': 484},
      {'loss': 7.42938756942749, 'step': 485},
      {'loss': 7.413568496704102, 'step': 486},
      {'loss': 7.398937702178955, 'step': 487},
      {'loss': 7.52539587020874, 'step': 488},
      {'loss': 7.429568767547607, 'step': 489},
      {'loss': 7.264672756195068, 'step': 490},
      {'loss': 6.875964641571045, 'step': 491},
      {'loss': 7.077130317687988, 'step': 492},
      {'loss': 7.291159629821777, 'step': 493},
      {'loss': 7.351440906524658, 'step': 494},
      {'loss': 7.305765151977539, 'step': 495},
      {'loss': 7.157251834869385, 'step': 496},
      {'loss': 6.864714622497559, 'step': 497},
      {'loss': 7.188262939453125, 'step': 498},
      {'loss': 6.995358943939209, 'step': 499},
      {'loss': 6.930678844451904, 'step': 500},
      {'loss': 6.9116692543029785, 'step': 501},
      {'loss': 7.005643844604492, 'step': 502},
      {'loss': 7.000782489776611, 'step': 503},
      {'loss': 7.164944171905518, 'step': 504},
      {'loss': 7.129715442657471, 'step': 505},
      {'loss': 6.745270729064941, 'step': 506},
      {'loss': 7.384077072143555, 'step': 507},
      {'loss': 6.983395576477051, 'step': 508},
      {'loss': 6.890662670135498, 'step': 509},
      {'loss': 7.051796913146973, 'step': 510},
      {'loss': 7.055161476135254, 'step': 511},
      {'loss': 6.967905044555664, 'step': 512},
      {'loss': 7.002407073974609, 'step': 513},
      {'loss': 7.159909248352051, 'step': 514},
      {'loss': 6.8897857666015625, 'step': 515},
      {'loss': 7.064925193786621, 'step': 516},
      {'loss': 7.024635314941406, 'step': 517},
      {'loss': 7.168274879455566, 'step': 518},
      {'loss': 7.466178894042969, 'step': 519},
      {'loss': 6.960180759429932, 'step': 520},
      {'loss': 6.981538772583008, 'step': 521},
      {'loss': 6.9790167808532715, 'step': 522},
      {'loss': 7.0309295654296875, 'step': 523},
      {'loss': 7.353662014007568, 'step': 524},
      {'loss': 7.095571994781494, 'step': 525},
      {'loss': 6.99553918838501, 'step': 526},
      {'loss': 6.716734409332275, 'step': 527},
      {'loss': 6.936665058135986, 'step': 528},
      {'loss': 7.109770774841309, 'step': 529},
      {'loss': 6.750192165374756, 'step': 530},
      {'loss': 7.002915859222412, 'step': 531},
      {'loss': 7.174258232116699, 'step': 532},
      {'loss': 6.924098491668701, 'step': 533},
      {'loss': 6.858094215393066, 'step': 534},
      {'loss': 7.413449287414551, 'step': 535},
      {'loss': 6.729828834533691, 'step': 536},
      {'loss': 7.076923847198486, 'step': 537},
      {'loss': 6.916792869567871, 'step': 538},
      {'loss': 6.8516669273376465, 'step': 539},
      {'loss': 7.062046051025391, 'step': 540},
      {'loss': 6.785090923309326, 'step': 541},
      {'loss': 6.851068496704102, 'step': 542},
      {'loss': 6.977838039398193, 'step': 543},
      {'loss': 6.923142910003662, 'step': 544},
      {'loss': 7.038519859313965, 'step': 545},
      {'loss': 7.238496780395508, 'step': 546},
      {'loss': 7.036983013153076, 'step': 547},
      {'loss': 6.6454057693481445, 'step': 548},
      {'loss': 6.9747395515441895, 'step': 549},
      {'loss': 7.136455059051514, 'step': 550},
      {'loss': 7.039889812469482, 'step': 551},
      {'loss': 7.228207588195801, 'step': 552},
      {'loss': 7.234391689300537, 'step': 553},
      {'loss': 6.964560508728027, 'step': 554},
      {'loss': 6.532079696655273, 'step': 555},
      {'loss': 6.896244525909424, 'step': 556},
      {'loss': 6.698338985443115, 'step': 557},
      {'loss': 6.694816589355469, 'step': 558},
      {'loss': 6.596003532409668, 'step': 559},
      {'loss': 6.8810014724731445, 'step': 560},
      {'loss': 6.929542541503906, 'step': 561},
      {'loss': 6.621035575866699, 'step': 562},
      {'loss': 6.912251949310303, 'step': 563},
      {'loss': 6.920406818389893, 'step': 564},
      {'loss': 6.879367828369141, 'step': 565},
      {'loss': 7.167364597320557, 'step': 566},
      {'loss': 6.806811332702637, 'step': 567},
      {'loss': 6.6051764488220215, 'step': 568},
      {'loss': 7.029492378234863, 'step': 569},
      {'loss': 6.9041337966918945, 'step': 570},
      {'loss': 6.625877857208252, 'step': 571},
      {'loss': 7.220470428466797, 'step': 572},
      {'loss': 6.735459327697754, 'step': 573},
      {'loss': 7.005105495452881, 'step': 574},
      {'loss': 6.871049404144287, 'step': 575},
      {'loss': 6.696956634521484, 'step': 576},
      {'loss': 6.719943046569824, 'step': 577},
      {'loss': 6.8283610343933105, 'step': 578},
      {'loss': 7.231015205383301, 'step': 579},
      {'loss': 6.931606769561768, 'step': 580},
      {'loss': 6.8368401527404785, 'step': 581},
      {'loss': 6.911436080932617, 'step': 582},
      {'loss': 6.775475978851318, 'step': 583},
      {'loss': 6.991641521453857, 'step': 584},
      {'loss': 6.621278762817383, 'step': 585},
      {'loss': 7.204102993011475, 'step': 586},
      {'loss': 6.732743263244629, 'step': 587},
      {'loss': 6.888396263122559, 'step': 588},
      {'loss': 7.009585857391357, 'step': 589},
      {'loss': 6.390641689300537, 'step': 590},
      {'loss': 6.643294334411621, 'step': 591},
      {'loss': 6.727651596069336, 'step': 592},
      {'loss': 6.9752607345581055, 'step': 593},
      {'loss': 6.951410293579102, 'step': 594},
      {'loss': 7.068047046661377, 'step': 595},
      {'loss': 7.102881908416748, 'step': 596},
      {'loss': 6.860496997833252, 'step': 597},
      {'loss': 6.662785053253174, 'step': 598},
      {'loss': 7.041903972625732, 'step': 599},
      {'loss': 6.653001308441162, 'step': 600},
      {'loss': 6.9331746101379395, 'step': 601},
      {'loss': 6.682084083557129, 'step': 602},
      {'loss': 6.604304313659668, 'step': 603},
      {'loss': 6.700200080871582, 'step': 604},
      {'loss': 7.158905029296875, 'step': 605},
      {'loss': 6.682844638824463, 'step': 606},
      {'loss': 6.3417768478393555, 'step': 607},
      {'loss': 6.803513526916504, 'step': 608},
      {'loss': 6.847768783569336, 'step': 609},
      {'loss': 6.493383407592773, 'step': 610},
      {'loss': 6.53223991394043, 'step': 611},
      {'loss': 7.1566314697265625, 'step': 612},
      {'loss': 6.543521404266357, 'step': 613},
      {'loss': 6.915842056274414, 'step': 614},
      {'loss': 6.681782245635986, 'step': 615},
      {'loss': 6.560512065887451, 'step': 616},
      {'loss': 6.918051242828369, 'step': 617},
      {'loss': 6.844791889190674, 'step': 618},
      {'loss': 6.642379283905029, 'step': 619},
      {'loss': 6.814577579498291, 'step': 620},
      {'loss': 6.301275253295898, 'step': 621},
      {'loss': 6.67906379699707, 'step': 622},
      {'loss': 6.723208904266357, 'step': 623},
      {'loss': 6.6230058670043945, 'step': 624},
      {'loss': 6.817347526550293, 'step': 625},
      {'loss': 6.498244762420654, 'step': 626},
      {'loss': 6.593464374542236, 'step': 627},
      {'loss': 6.467695713043213, 'step': 628},
      {'loss': 6.3463969230651855, 'step': 629},
      {'loss': 6.580927848815918, 'step': 630},
      {'loss': 6.7846832275390625, 'step': 631},
      {'loss': 6.631278991699219, 'step': 632},
      {'loss': 7.048778533935547, 'step': 633},
      {'loss': 6.388098239898682, 'step': 634},
      {'loss': 6.798262119293213, 'step': 635},
      {'loss': 6.878608703613281, 'step': 636},
      {'loss': 6.206899642944336, 'step': 637},
      {'loss': 6.444216728210449, 'step': 638},
      {'loss': 6.49919319152832, 'step': 639},
      {'loss': 6.683052062988281, 'step': 640},
      {'loss': 6.505168914794922, 'step': 641},
      {'loss': 6.582919597625732, 'step': 642},
      {'loss': 6.661754131317139, 'step': 643},
      {'loss': 6.645831108093262, 'step': 644},
      {'loss': 6.693548679351807, 'step': 645},
      {'loss': 6.753129959106445, 'step': 646},
      {'loss': 6.851592063903809, 'step': 647},
      {'loss': 6.398972511291504, 'step': 648},
      {'loss': 6.512589454650879, 'step': 649},
      {'loss': 6.56635856628418, 'step': 650},
      {'loss': 6.573788166046143, 'step': 651},
      {'loss': 6.675108432769775, 'step': 652},
      {'loss': 6.4293975830078125, 'step': 653},
      {'loss': 6.469366550445557, 'step': 654},
      {'loss': 6.380842685699463, 'step': 655},
      {'loss': 6.6808671951293945, 'step': 656},
      {'loss': 6.868982315063477, 'step': 657},
      {'loss': 6.6302642822265625, 'step': 658},
      {'loss': 6.670416831970215, 'step': 659},
      {'loss': 6.539815902709961, 'step': 660},
      {'loss': 6.861855506896973, 'step': 661},
      {'loss': 6.464282989501953, 'step': 662},
      {'loss': 6.387969970703125, 'step': 663},
      {'loss': 6.6908278465271, 'step': 664},
      {'loss': 6.82158088684082, 'step': 665},
      {'loss': 6.363522529602051, 'step': 666},
      {'loss': 6.686793327331543, 'step': 667},
      {'loss': 6.44807767868042, 'step': 668},
      {'loss': 6.294849872589111, 'step': 669},
      {'loss': 6.235434532165527, 'step': 670},
      {'loss': 6.574227809906006, 'step': 671},
      {'loss': 5.937483310699463, 'step': 672},
      {'loss': 6.670145034790039, 'step': 673},
      {'loss': 6.309910774230957, 'step': 674},
      {'loss': 6.672595977783203, 'step': 675},
      {'loss': 6.67143440246582, 'step': 676},
      {'loss': 7.058235168457031, 'step': 677},
      {'loss': 6.337688446044922, 'step': 678},
      {'loss': 6.464450836181641, 'step': 679},
      {'loss': 6.585697174072266, 'step': 680},
      {'loss': 6.420019626617432, 'step': 681},
      {'loss': 7.015267372131348, 'step': 682},
      {'loss': 6.811161518096924, 'step': 683},
      {'loss': 6.418643951416016, 'step': 684},
      {'loss': 6.785398960113525, 'step': 685},
      {'loss': 6.375899314880371, 'step': 686},
      {'loss': 6.245916843414307, 'step': 687},
      {'loss': 6.400662899017334, 'step': 688},
      {'loss': 6.485900402069092, 'step': 689},
      {'loss': 6.231479167938232, 'step': 690},
      {'loss': 6.030235290527344, 'step': 691},
      {'loss': 6.598134994506836, 'step': 692},
      {'loss': 6.817981719970703, 'step': 693},
      {'loss': 6.151371479034424, 'step': 694},
      {'loss': 6.866644382476807, 'step': 695},
      {'loss': 6.287429332733154, 'step': 696},
      {'loss': 6.458990097045898, 'step': 697},
      {'loss': 6.910857677459717, 'step': 698},
      {'loss': 6.175490856170654, 'step': 699},
      {'loss': 6.762678623199463, 'step': 700},
      {'loss': 6.306063175201416, 'step': 701},
      {'loss': 6.234595775604248, 'step': 702},
      {'loss': 6.559793472290039, 'step': 703},
      {'loss': 6.181817531585693, 'step': 704},
      {'loss': 6.422220706939697, 'step': 705},
      {'loss': 6.413985252380371, 'step': 706},
      {'loss': 6.2502665519714355, 'step': 707},
      {'loss': 6.423414707183838, 'step': 708},
      {'loss': 6.146265983581543, 'step': 709},
      {'loss': 6.04506254196167, 'step': 710},
      {'loss': 6.192164897918701, 'step': 711},
      {'loss': 6.3821234703063965, 'step': 712},
      {'loss': 6.74046516418457, 'step': 713},
      {'loss': 6.4006452560424805, 'step': 714},
      {'loss': 6.401009559631348, 'step': 715},
      {'loss': 6.119245529174805, 'step': 716},
      {'loss': 6.211571216583252, 'step': 717},
      {'loss': 6.302793979644775, 'step': 718},
      {'loss': 6.589457988739014, 'step': 719},
      {'loss': 6.220929145812988, 'step': 720},
      {'loss': 6.622432231903076, 'step': 721},
      {'loss': 6.260295867919922, 'step': 722},
      {'loss': 6.410425186157227, 'step': 723},
      {'loss': 6.017321586608887, 'step': 724},
      {'loss': 6.604228973388672, 'step': 725},
      {'loss': 6.556307315826416, 'step': 726},
      {'loss': 6.362915515899658, 'step': 727},
      {'loss': 6.591972827911377, 'step': 728},
      {'loss': 6.242903709411621, 'step': 729},
      {'loss': 6.133462905883789, 'step': 730},
      {'loss': 6.620084762573242, 'step': 731},
      {'loss': 6.445712566375732, 'step': 732},
      {'loss': 6.673415660858154, 'step': 733},
      {'loss': 6.4357523918151855, 'step': 734},
      {'loss': 6.277858257293701, 'step': 735},
      {'loss': 6.11099910736084, 'step': 736},
      {'loss': 5.989086627960205, 'step': 737},
      {'loss': 6.280193328857422, 'step': 738},
      {'loss': 6.052328586578369, 'step': 739},
      {'loss': 6.216166973114014, 'step': 740},
      {'loss': 6.426962375640869, 'step': 741},
      {'loss': 6.064672946929932, 'step': 742},
      {'loss': 5.979720115661621, 'step': 743},
      {'loss': 6.442633152008057, 'step': 744},
      {'loss': 5.977102279663086, 'step': 745},
      {'loss': 6.067193508148193, 'step': 746},
      {'loss': 5.80436372756958, 'step': 747},
      {'loss': 6.289169788360596, 'step': 748},
      {'loss': 6.385468482971191, 'step': 749},
      {'loss': 6.324035167694092, 'step': 750},
      {'loss': 6.517641067504883, 'step': 751},
      {'loss': 6.252870082855225, 'step': 752},
      {'loss': 5.905908584594727, 'step': 753},
      {'loss': 6.183718681335449, 'step': 754},
      {'loss': 6.179794788360596, 'step': 755},
      {'loss': 6.349639892578125, 'step': 756},
      {'loss': 6.641066551208496, 'step': 757},
      {'loss': 6.4225921630859375, 'step': 758},
      {'loss': 6.158094882965088, 'step': 759},
      {'loss': 6.290485382080078, 'step': 760},
      {'loss': 6.1289544105529785, 'step': 761},
      {'loss': 6.061132907867432, 'step': 762},
      {'loss': 6.177314758300781, 'step': 763},
      {'loss': 6.055022239685059, 'step': 764},
      {'loss': 6.397853851318359, 'step': 765},
      {'loss': 6.090449333190918, 'step': 766},
      {'loss': 6.319674015045166, 'step': 767},
      {'loss': 6.3961005210876465, 'step': 768},
      {'loss': 6.475203990936279, 'step': 769},
      {'loss': 6.230829238891602, 'step': 770},
      {'loss': 6.18487024307251, 'step': 771},
      {'loss': 5.79786491394043, 'step': 772},
      {'loss': 5.824963092803955, 'step': 773},
      {'loss': 5.739922046661377, 'step': 774},
      {'loss': 6.074657440185547, 'step': 775},
      {'loss': 6.092113494873047, 'step': 776},
      {'loss': 6.5202460289001465, 'step': 777},
      {'loss': 5.902870178222656, 'step': 778},
      {'loss': 5.97307825088501, 'step': 779},
      {'loss': 6.207627296447754, 'step': 780},
      {'loss': 6.485926628112793, 'step': 781},
      {'loss': 6.567532539367676, 'step': 782},
      {'loss': 6.365774154663086, 'step': 783},
      {'loss': 6.111070156097412, 'step': 784},
      {'loss': 5.809693336486816, 'step': 785},
      {'loss': 5.6428399085998535, 'step': 786},
      {'loss': 6.147626876831055, 'step': 787},
      {'loss': 5.9814276695251465, 'step': 788},
      {'loss': 6.297543525695801, 'step': 789},
      {'loss': 6.170338153839111, 'step': 790},
      {'loss': 6.398639678955078, 'step': 791},
      {'loss': 6.274231910705566, 'step': 792},
      {'loss': 6.36652135848999, 'step': 793},
      {'loss': 5.930144786834717, 'step': 794},
      {'loss': 6.171633720397949, 'step': 795},
      {'loss': 6.131617546081543, 'step': 796},
      {'loss': 6.3817524909973145, 'step': 797},
      {'loss': 6.184343338012695, 'step': 798},
      {'loss': 6.230269432067871, 'step': 799},
      {'loss': 6.179476737976074, 'step': 800},
      {'loss': 6.41139030456543, 'step': 801},
      {'loss': 6.008386611938477, 'step': 802},
      {'loss': 6.165910243988037, 'step': 803},
      {'loss': 6.1855878829956055, 'step': 804},
      {'loss': 5.893813133239746, 'step': 805},
      {'loss': 6.154047012329102, 'step': 806},
      {'loss': 6.244041442871094, 'step': 807},
      {'loss': 6.213301181793213, 'step': 808},
      {'loss': 6.344395637512207, 'step': 809},
      {'loss': 6.192485809326172, 'step': 810},
      {'loss': 6.063512802124023, 'step': 811},
      {'loss': 6.23832368850708, 'step': 812},
      {'loss': 6.149255275726318, 'step': 813},
      {'loss': 6.224613189697266, 'step': 814},
      {'loss': 6.104172706604004, 'step': 815},
      {'loss': 6.038175582885742, 'step': 816},
      {'loss': 6.170835018157959, 'step': 817},
      {'loss': 5.6534504890441895, 'step': 818},
      {'loss': 5.556952953338623, 'step': 819},
      {'loss': 5.765564441680908, 'step': 820},
      {'loss': 6.162184238433838, 'step': 821},
      {'loss': 6.334968566894531, 'step': 822},
      {'loss': 6.043604373931885, 'step': 823},
      {'loss': 5.505506992340088, 'step': 824},
      {'loss': 5.809406757354736, 'step': 825},
      {'loss': 5.731060981750488, 'step': 826},
      {'loss': 5.781925678253174, 'step': 827},
      {'loss': 6.456454753875732, 'step': 828},
      {'loss': 5.5982985496521, 'step': 829},
      {'loss': 6.007455825805664, 'step': 830},
      {'loss': 6.023557186126709, 'step': 831},
      {'loss': 5.756533145904541, 'step': 832},
      {'loss': 6.216346740722656, 'step': 833},
      {'loss': 5.977278709411621, 'step': 834},
      {'loss': 6.46477746963501, 'step': 835},
      {'loss': 6.132097244262695, 'step': 836},
      {'loss': 6.160898208618164, 'step': 837},
      {'loss': 5.840658187866211, 'step': 838},
      {'loss': 5.917392730712891, 'step': 839},
      {'loss': 6.022525310516357, 'step': 840},
      {'loss': 5.649994850158691, 'step': 841},
      {'loss': 6.224531650543213, 'step': 842},
      {'loss': 6.067052364349365, 'step': 843},
      {'loss': 5.698977470397949, 'step': 844},
      {'loss': 5.919789791107178, 'step': 845},
      {'loss': 6.412214279174805, 'step': 846},
      {'loss': 6.1434006690979, 'step': 847},
      {'loss': 5.587151050567627, 'step': 848},
      {'loss': 5.95936393737793, 'step': 849},
      {'loss': 5.577254295349121, 'step': 850},
      {'loss': 5.656014919281006, 'step': 851},
      {'loss': 5.839982509613037, 'step': 852},
      {'loss': 5.610649585723877, 'step': 853},
      {'loss': 6.198298454284668, 'step': 854},
      {'loss': 6.205420017242432, 'step': 855},
      {'loss': 6.007063865661621, 'step': 856},
      {'loss': 5.95534610748291, 'step': 857},
      {'loss': 6.115119457244873, 'step': 858},
      {'loss': 5.848546504974365, 'step': 859},
      {'loss': 5.4353814125061035, 'step': 860},
      {'loss': 6.194253921508789, 'step': 861},
      {'loss': 5.886451244354248, 'step': 862},
      {'loss': 6.084601402282715, 'step': 863},
      {'loss': 6.205596923828125, 'step': 864},
      {'loss': 5.713630676269531, 'step': 865},
      {'loss': 6.155368804931641, 'step': 866},
      {'loss': 5.987909317016602, 'step': 867},
      {'loss': 5.949391841888428, 'step': 868},
      {'loss': 5.897697925567627, 'step': 869},
      {'loss': 6.168738842010498, 'step': 870},
      {'loss': 5.7707905769348145, 'step': 871},
      {'loss': 5.584583759307861, 'step': 872},
      {'loss': 5.786444187164307, 'step': 873},
      {'loss': 5.585447788238525, 'step': 874},
      {'loss': 5.662837982177734, 'step': 875},
      {'loss': 5.712212562561035, 'step': 876},
      {'loss': 5.970769882202148, 'step': 877},
      {'loss': 5.702062129974365, 'step': 878},
      {'loss': 6.107863903045654, 'step': 879},
      {'loss': 6.066924095153809, 'step': 880},
      {'loss': 5.663297176361084, 'step': 881},
      {'loss': 5.531920433044434, 'step': 882},
      {'loss': 5.892160892486572, 'step': 883},
      {'loss': 6.287512302398682, 'step': 884},
      {'loss': 5.848054885864258, 'step': 885},
      {'loss': 5.627744674682617, 'step': 886},
      {'loss': 5.806376934051514, 'step': 887},
      {'loss': 6.096044540405273, 'step': 888},
      {'loss': 6.111745834350586, 'step': 889},
      {'loss': 5.776045322418213, 'step': 890},
      {'loss': 6.22074031829834, 'step': 891},
      {'loss': 5.90162467956543, 'step': 892},
      {'loss': 5.47128438949585, 'step': 893},
      {'loss': 5.359961032867432, 'step': 894},
      {'loss': 5.420084476470947, 'step': 895},
      {'loss': 5.704773902893066, 'step': 896},
      {'loss': 6.192122936248779, 'step': 897},
      {'loss': 5.302196502685547, 'step': 898},
      {'loss': 6.1650390625, 'step': 899},
      {'loss': 5.282564640045166, 'step': 900},
      {'loss': 5.333053112030029, 'step': 901},
      {'loss': 5.954625129699707, 'step': 902},
      {'loss': 5.953242301940918, 'step': 903},
      {'loss': 6.22835636138916, 'step': 904},
      {'loss': 5.403650283813477, 'step': 905},
      {'loss': 5.454579830169678, 'step': 906},
      {'loss': 6.035684585571289, 'step': 907},
      {'loss': 5.9421706199646, 'step': 908},
      {'loss': 5.840601444244385, 'step': 909},
      {'loss': 5.540421962738037, 'step': 910},
      {'loss': 5.699370384216309, 'step': 911},
      {'loss': 6.241888523101807, 'step': 912},
      {'loss': 5.99316930770874, 'step': 913},
      {'loss': 5.508965969085693, 'step': 914},
      {'loss': 6.035554885864258, 'step': 915},
      {'loss': 5.975940227508545, 'step': 916},
      {'loss': 5.44652795791626, 'step': 917},
      {'loss': 5.783436298370361, 'step': 918},
      {'loss': 5.651610851287842, 'step': 919},
      {'loss': 6.027553558349609, 'step': 920},
      {'loss': 5.427163124084473, 'step': 921},
      {'loss': 6.176769256591797, 'step': 922},
      {'loss': 5.752948760986328, 'step': 923},
      {'loss': 5.620815277099609, 'step': 924},
      {'loss': 6.114317417144775, 'step': 925},
      {'loss': 5.740330696105957, 'step': 926},
      {'loss': 5.570556163787842, 'step': 927},
      {'loss': 5.751370906829834, 'step': 928},
      {'loss': 6.004382610321045, 'step': 929},
      {'loss': 5.783803462982178, 'step': 930},
      {'loss': 5.746761322021484, 'step': 931},
      {'loss': 6.198850154876709, 'step': 932},
      {'loss': 5.722126007080078, 'step': 933},
      {'loss': 6.162303447723389, 'step': 934},
      {'loss': 5.971364974975586, 'step': 935},
      {'loss': 6.06383752822876, 'step': 936},
      {'loss': 5.631424427032471, 'step': 937},
      {'loss': 5.849540710449219, 'step': 938},
      {'loss': 6.007293701171875, 'step': 939},
      {'loss': 5.665019512176514, 'step': 940},
      {'loss': 5.725893497467041, 'step': 941},
      {'loss': 5.521500587463379, 'step': 942},
      {'loss': 5.599538803100586, 'step': 943},
      {'loss': 6.301260948181152, 'step': 944},
      {'loss': 6.1965436935424805, 'step': 945},
      {'loss': 5.441428184509277, 'step': 946},
      {'loss': 5.766757011413574, 'step': 947},
      {'loss': 5.58862829208374, 'step': 948},
      {'loss': 6.016271114349365, 'step': 949},
      {'loss': 6.092831134796143, 'step': 950},
      {'loss': 5.5640549659729, 'step': 951},
      {'loss': 5.865227699279785, 'step': 952},
      {'loss': 5.768653869628906, 'step': 953},
      {'loss': 5.741983890533447, 'step': 954},
      {'loss': 6.127964496612549, 'step': 955},
      {'loss': 5.589209079742432, 'step': 956},
      {'loss': 5.7384772300720215, 'step': 957},
      {'loss': 5.333614349365234, 'step': 958},
      {'loss': 5.813440799713135, 'step': 959},
      {'loss': 5.3871588706970215, 'step': 960},
      {'loss': 5.936622142791748, 'step': 961},
      {'loss': 5.299325466156006, 'step': 962},
      {'loss': 6.04257869720459, 'step': 963},
      {'loss': 5.441106796264648, 'step': 964},
      {'loss': 6.040995121002197, 'step': 965},
      {'loss': 5.636991024017334, 'step': 966},
      {'loss': 5.717286109924316, 'step': 967},
      {'loss': 5.76033353805542, 'step': 968},
      {'loss': 5.318562984466553, 'step': 969},
      {'loss': 6.139246463775635, 'step': 970},
      {'loss': 5.894296169281006, 'step': 971},
      {'loss': 6.329341411590576, 'step': 972},
      {'loss': 5.847519874572754, 'step': 973},
      {'loss': 6.109082221984863, 'step': 974},
      {'loss': 5.4984331130981445, 'step': 975},
      {'loss': 5.820492744445801, 'step': 976},
      {'loss': 5.210939884185791, 'step': 977},
      {'loss': 5.375313758850098, 'step': 978},
      {'loss': 5.687919616699219, 'step': 979},
      {'loss': 5.859033107757568, 'step': 980},
      {'loss': 5.357014179229736, 'step': 981},
      {'loss': 5.904397010803223, 'step': 982},
      {'loss': 5.5888495445251465, 'step': 983},
      {'loss': 5.191726207733154, 'step': 984},
      {'loss': 6.00544548034668, 'step': 985},
      {'loss': 5.513291358947754, 'step': 986},
      {'loss': 5.283057689666748, 'step': 987},
      {'loss': 5.908671855926514, 'step': 988},
      {'loss': 5.706696033477783, 'step': 989},
      {'loss': 5.8201751708984375, 'step': 990},
      {'loss': 5.428075790405273, 'step': 991},
      {'loss': 5.835113048553467, 'step': 992},
      {'loss': 5.517333984375, 'step': 993},
      {'loss': 5.341905117034912, 'step': 994},
      {'loss': 5.483656883239746, 'step': 995},
      {'loss': 5.83479118347168, 'step': 996},
      {'loss': 5.608740329742432, 'step': 997},
      {'loss': 5.4710493087768555, 'step': 998},
      {'loss': 5.999269485473633, 'step': 999},
      {'loss': 5.313358306884766, 'step': 1000},
      ...],
     'val': [{'loss': 7.091875350475311, 'step': 500},
      {'loss': 5.678749084472656, 'step': 1000},
      {'loss': 5.3444198846817015, 'step': 1500},
      {'loss': 4.978909265995026, 'step': 2000},
      {'loss': 4.755400604009628, 'step': 2500},
      {'loss': 4.528371220827102, 'step': 3000},
      {'loss': 4.483922469615936, 'step': 3500},
      {'loss': 4.284839218854904, 'step': 4000},
      {'loss': 4.183202081918717, 'step': 4500},
      {'loss': 4.083735638856888, 'step': 5000},
      {'loss': 4.014825022220611, 'step': 5500},
      {'loss': 3.9412540435791015, 'step': 6000},
      {'loss': 3.8768269419670105, 'step': 6500},
      {'loss': 3.841353738307953, 'step': 7000},
      {'loss': 3.7859415888786314, 'step': 7500},
      {'loss': 3.7475524723529814, 'step': 8000},
      {'loss': 3.696993100643158, 'step': 8500},
      {'loss': 3.667452424764633, 'step': 9000},
      {'loss': 3.636081737279892, 'step': 9500},
      {'loss': 3.6453403770923614, 'step': 10000},
      {'loss': 3.589928412437439, 'step': 10500},
      {'loss': 3.5882970333099364, 'step': 11000},
      {'loss': 3.5510027587413786, 'step': 11500},
      {'loss': 3.5370976269245147, 'step': 12000},
      {'loss': 3.5140486121177674, 'step': 12500},
      {'loss': 3.5072770535945894, 'step': 13000},
      {'loss': 3.4780659437179566, 'step': 13500},
      {'loss': 3.4552790403366087, 'step': 14000},
      {'loss': 3.4570499837398527, 'step': 14500},
      {'loss': 3.437154144048691, 'step': 15000},
      {'loss': 3.4098286628723145, 'step': 15500},
      {'loss': 3.411135971546173, 'step': 16000},
      {'loss': 3.386381435394287, 'step': 16500},
      {'loss': 3.3924562692642213, 'step': 17000},
      {'loss': 3.370198208093643, 'step': 17500},
      {'loss': 3.3717746913433073, 'step': 18000},
      {'loss': 3.353752779960632, 'step': 18500},
      {'loss': 3.3413534998893737, 'step': 19000},
      {'loss': 3.334506767988205, 'step': 19500},
      {'loss': 3.326050990819931, 'step': 20000},
      {'loss': 3.334199404716492, 'step': 20500},
      {'loss': 3.32095645070076, 'step': 21000},
      {'loss': 3.32389372587204, 'step': 21500},
      {'loss': 3.306799793243408, 'step': 22000},
      {'loss': 3.3067815959453584, 'step': 22500},
      {'loss': 3.3025163888931273, 'step': 23000},
      {'loss': 3.301250070333481, 'step': 23500},
      {'loss': 3.300953286886215, 'step': 24000},
      {'loss': 3.2883458197116853, 'step': 24500},
      {'loss': 3.2873795211315153, 'step': 25000},
      {'loss': 3.2881278693675995, 'step': 25500},
      {'loss': 3.288409101963043, 'step': 26000},
      {'loss': 3.2851992785930633, 'step': 26500},
      {'loss': 3.283895468711853, 'step': 27000},
      {'loss': 3.2858198821544646, 'step': 27500},
      {'loss': 3.2887156426906587, 'step': 28000},
      {'loss': 3.2892450511455538, 'step': 28500},
      {'loss': 3.2848232209682466, 'step': 29000},
      {'loss': 3.2837136507034304, 'step': 29500},
      {'loss': 3.286137694120407, 'step': 30000},
      {'loss': 3.284578424692154, 'step': 30500},
      {'loss': 3.2906786620616915, 'step': 31000},
      {'loss': 3.284909850358963, 'step': 31500},
      {'loss': 3.287942445278168, 'step': 32000}]}



## 推理

- 翻译项目的评估指标一般是BLEU4，感兴趣的同学自行了解并实现
- 接下来进行翻译推理，并作出注意力的热度图


```python
!pip install Cython  # if failed to install fastBPE, try this line
!pip install fastBPE #分词使用
# 在 Windows 系统上并没有 sys/mman.h 文件
```

    Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
    Requirement already satisfied: Cython in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (3.0.11)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m[33m
    [0mLooking in indexes: https://mirrors.aliyun.com/pypi/simple/
    Requirement already satisfied: fastBPE in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (0.1.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m[33m
    [0m


```python
exp_name
```




    'translate-transformer-not-share'




```python
!ls checkpoints/translate-transformer-not-share -l
```

    total 231188
    -rw-r--r-- 1 root root 236736459  8月  6 23:39 best.ckpt



```python
import torch

state_dict = torch.load(f"checkpoints/translate-transformer-not-share/best.ckpt", map_location=device)

# state_dict1 = torch.load("epoch125-step132426.ckpt", map_location="cpu")
# state_dict = state_dict1["state_dict"]

# update keys by dropping `model`
# for key in list(state_dict):
#     state_dict[key.replace("model.", "")] = state_dict.pop(key)

```


```python
!pip install nltk
```

    Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
    Requirement already satisfied: nltk in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (3.9.1)
    Requirement already satisfied: click in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (from nltk) (8.1.7)
    Requirement already satisfied: joblib in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (from nltk) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (from nltk) (2024.11.6)
    Requirement already satisfied: tqdm in /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages (from nltk) (4.67.1)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m[33m
    [0m


```python
!rm -r wmt16/.cache
```


```python
tokenizer.decode([[   5,   16,    6,   23,  150,   80, 8248,   35,  232,    4,    3]])
```




    ['a man in an orange hat starring at something .']




```python
from nltk.translate.bleu_score import sentence_bleu
# load checkpoints
model = TransformerModel(config)
model.load_state_dict(state_dict)

loss_fct = CrossEntropyWithPadding(config)
# from dataset import LangPairDataset
test_ds = LangPairDataset("test", max_length=128, data_dir="./wmt16")
test_dl = DataLoader(test_ds, batch_size=1, collate_fn=partial(collate_fct, tokenizer=tokenizer))

model = model.to(device)
model.eval()
collect = {}
loss_collect = []

predictions = []
answers = []
# 初始化BLEU分数列表
bleu_scores = []
for idx, batch in tqdm(enumerate(test_dl)):
    encoder_inputs = batch["encoder_inputs"]
    encoder_inputs_mask = batch["encoder_inputs_mask"]
    decoder_inputs = batch["decoder_inputs"]
    decoder_labels = batch["decoder_labels"]
    # print(decoder_labels.cpu())
    # decoder_labels1=tokenizer.decode(decoder_labels.cpu().numpy())
    # print(decoder_labels1)
    # 前向计算
    outputs = model(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        encoder_inputs_mask=encoder_inputs_mask
        )
    loss = loss_fct(outputs.logits, decoder_labels)         # 验证集损失

    # print(outputs.logits.shape, decoder_labels.shape)

    # loss = loss_fct(outputs.logits[:, :decoder_labels.shape[1]], decoder_labels)         # 验证集损失
    # outputs = model.infer(encoder_inputs=encoder_inputs)
    # print(outputs.logits.shape)
    preds = outputs.logits.argmax(dim=-1) # 预测结果，[1,seq_len]
    # print(preds.shape)
    #把preds转为英文单词
    preds = tokenizer.decode(preds.cpu().numpy()) #['预测句子']
    # predictions.append(preds)
    # print(preds)
    #把decoder_labels转为英文单词
    decoder_labels = tokenizer.decode(decoder_labels.cpu().numpy()) #['标签句子']
    # answers.append(decoder_labels)
    # print(decoder_labels)
    belu=sentence_bleu([decoder_labels[0].split()],preds[0].split(),weights=(1, 0, 0, 0))
    bleu_scores.append(belu)
    collect[idx] = {"loss": loss.item(), "src_inputs": encoder_inputs, "trg_inputs": decoder_inputs, "mask": encoder_inputs_mask, "trg_labels": decoder_labels, "preds": preds}
    loss_collect.append(loss.item())
    # break

# sort collect by value
collect = sorted(collect.items(), key=lambda x: x[1]["loss"])
print(f"testing loss: {np.array(loss_collect).mean()}")
sum(bleu_scores) / len(bleu_scores)
```

    save cache to wmt16/.cache/de2en_test_128.npy



    0it [00:00, ?it/s]


    /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages/nltk/translate/bleu_score.py:577: UserWarning: 
    The hypothesis contains 0 counts of 4-gram overlaps.
    Therefore the BLEU score evaluates to 0, independently of
    how many N-gram overlaps of lower order it contains.
    Consider using lower n-gram order or use SmoothingFunction()
      warnings.warn(_msg)
    /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages/nltk/translate/bleu_score.py:577: UserWarning: 
    The hypothesis contains 0 counts of 3-gram overlaps.
    Therefore the BLEU score evaluates to 0, independently of
    how many N-gram overlaps of lower order it contains.
    Consider using lower n-gram order or use SmoothingFunction()
      warnings.warn(_msg)
    /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages/nltk/translate/bleu_score.py:577: UserWarning: 
    The hypothesis contains 0 counts of 2-gram overlaps.
    Therefore the BLEU score evaluates to 0, independently of
    how many N-gram overlaps of lower order it contains.
    Consider using lower n-gram order or use SmoothingFunction()
      warnings.warn(_msg)


    testing loss: 3.1873543899096743





    0.5897835772081111




```python
import re
from fastBPE import fastBPE
from sacremoses import MosesDetokenizer, MosesTokenizer

# `MosesTokenizer` 和 `MosesDetokenizer` 是来自 `sacremoses` 库的工具，用于自然语言处理中的分词（Tokenization）和去标记化（Detokenization）。这些工具主要用于对文本进行预处理和后处理，通常在处理自然语言处理任务时会用到。
#
# ### MosesTokenizer：
# - **作用**：将原始文本分割成单词和标点符号。
# - **特点**：基于 Moses 翻译工具中使用的分词方法。
# - **功能**：
#   - 将句子分割成单词和标点符号。
#   - 处理缩写、连字符、标点等特殊情况。
#   - 对文本进行标记化，方便后续处理。
#
# ### MosesDetokenizer：
# - **作用**：将分词后的文本重新组合成原始的句子。
# - **特点**：用于对分词后的文本进行还原，使其恢复为可读的句子形式。
# - **功能**：
#   - 将分词后的单词和标点符号重新组合成句子。
#   - 处理分词后的标点、缩写等情况，使得结果更加自然和可读。
#
# 这些工具通常在文本预处理和后处理过程中使用，对输入的文本进行标记化和去标记化，是一种常用的处理方式。在自然语言处理任务中，对文本进行正确的分词和还原是很重要的，而 `MosesTokenizer` 和 `MosesDetokenizer` 提供了方便、高效的工具来处理这些任务。

class Translator:
    def __init__(self, model, src_tokenizer, trg_tokenizer):
        self.bpe = fastBPE("./wmt16/bpe.10000", "./wmt16/vocab")
        self.mose_tokenizer = MosesTokenizer(lang="de")
        self.mose_detokenizer = MosesDetokenizer(lang="en")
        self.model = model
        self.model.eval()
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.pattern = re.compile(r'(@@ )|(@@ ?$)')

    def draw_attention_map(self, attn_scores, cross_attn_scores, src_words_list, trg_words_list):
        """绘制注意力热力图
        attn_scores (numpy.ndarray): 表示自注意力机制（self-attention）分数。
        cross_attn_scores (numpy.ndarray): 表示交叉注意力机制的注意力分数。
        src_words_list (list): 源语言句子的单词列表。
        trg_words_list (list): 目标语言句子的单词列表。
        """
        assert len(attn_scores.shape) == 3, "attn_scores shape should be " \
            f"[num heads, target sequence length, target sequence length], but got {attn_scores.shape}"
        attn_scores = attn_scores[:, :len(trg_words_list), :len(trg_words_list)]

        assert len(cross_attn_scores.shape) == 3, "attn_scores shape should be " \
            f"[num heads, target sequence length, source sequence length], but got {cross_attn_scores.shape}"
        cross_attn_scores = cross_attn_scores[:, :len(trg_words_list), :len(src_words_list)]

        num_heads, trg_len, src_len = cross_attn_scores.shape

        fig = plt.figure(figsize=(10, 5), constrained_layout=True) # constrained_layout=True 自动调整子图参数，使之填充整个图像区域
        grid = plt.GridSpec(trg_len, trg_len + src_len, wspace=0.1, hspace=0.1)# wspace,hspace 控制子图之间的间距
        #下面是attn_scores的热力图
        self_map = fig.add_subplot(grid[:,:trg_len]) #  添加子图
        self_map.matshow(attn_scores.mean(dim=0), cmap='viridis') # 绘制热力图，cmap表示颜色,dim=0表示对第0维求均值
        self_map.set_yticks(range(trg_len), trg_words_list, fontsize=10)
        self_map.set_xticks(range(trg_len), ["[BOS]"] + trg_words_list[:-1], rotation=90)
        #下面是cross_attn_scores的热力图
        cross_map = fig.add_subplot(grid[:, trg_len:])
        cross_map.matshow(cross_attn_scores.mean(dim=0), cmap='viridis')
        cross_map.set_yticks(range(trg_len), [], fontsize=6)
        cross_map.set_xticks(range(src_len), src_words_list, rotation=90)

        plt.show()

    def draw_attention_maps(self, attn_scores, cross_attn_scores, src_words_list, trg_words_list, heads_list):
        """绘制注意力热力图

        Args:
            - scores (numpy.ndarray): shape = [source sequence length, target sequence length]
        """
        assert len(attn_scores.shape) == 3, "attn_scores shape should be " \
            f"[num heads, target sequence length, target sequence length], but got {attn_scores.shape}"
        attn_scores = attn_scores[:, :len(trg_words_list), :len(trg_words_list)]

        assert len(cross_attn_scores.shape) == 3, "attn_scores shape should be " \
            f"[num heads, target sequence length, source sequence length], but got {cross_attn_scores.shape}"
        cross_attn_scores = cross_attn_scores[:, :len(trg_words_list), :len(src_words_list)]
        # cross_attn_scores = cross_attn_scores[:, :len(src_words_list), :len(src_words_list)]

        num_heads, trg_len, src_len = cross_attn_scores.shape
        fig, axes = plt.subplots(2, len(heads_list), figsize=(5 * len(heads_list), 10))
        for i, heads_idx in enumerate(heads_list):
            axes[0, i].matshow(attn_scores[heads_idx], cmap='viridis')
            axes[0, i].set_yticks(range(trg_len), trg_words_list)
            axes[0, i].set_xticks(range(trg_len), ["[BOS]"] + trg_words_list[:-1], rotation=90)
            axes[0, i].set_title(f"head {heads_idx}")
            axes[1, i].matshow(cross_attn_scores[heads_idx], cmap='viridis')
            axes[1, i].set_yticks(range(trg_len), trg_words_list)
            axes[1, i].set_xticks(range(src_len), src_words_list, rotation=90)
            axes[1, i].set_title(f"head {heads_idx}")

        plt.show()


    def __call__(self, sentence_list, heads_list=None, layer_idx=-1):
        # 将输入句子列表转换为小写，并使用 MosesTokenizer 进行分词处理。
        sentence_list = [" ".join(self.mose_tokenizer.tokenize(s.lower())) for s in sentence_list]
        # 将分词后的结果进行 BPE 编码，得到 tokens_list。
        tokens_list = [s.split() for s in self.bpe.apply(sentence_list)]
        # 使用 src_tokenizer 对 tokens_list 进行编码，同时添加起始标记 ([BOS]) 和结束标记 ([EOS])。
        encoder_input, attn_mask = self.src_tokenizer.encode(
            tokens_list,
            add_bos=True,
            add_eos=True,
            return_mask=True,
            )
        encoder_input = torch.Tensor(encoder_input).to(dtype=torch.int64)
        # 使用模型的 infer 方法对编码器输入进行推理，得到输出结果 outputs
        outputs = model.infer(encoder_inputs=encoder_input, encoder_inputs_mask=attn_mask)

        preds = outputs.preds.numpy()
        # 使用目标语言的 trg_tokenizer 对预测序列进行解码，得到解码后的目标语言句子列表 trg_decoded。
        trg_decoded = self.trg_tokenizer.decode(preds, split=True, remove_eos=False, remove_bos=False, remove_pad=False)
        # 使用源语言的 src_tokenizer 对编码器输入进行解码，得到解码后的源语言句子列表 src_decoded。为下面绘制热力图做准备。
        src_decoded = self.src_tokenizer.decode(
            encoder_input.numpy(),
            split=True,
            remove_bos=False,
            remove_eos=False
            )

        # post processed attn scores
        # outputs.decoder_attentions[-1]  # the last layer of self-attention scores

        # draw the attention map of the last decoder block
        for attn_score, cross_attn_score, src, trg in zip(
            outputs.decoder_self_attn_scores[layer_idx], outputs.decoder_cross_attn_scores[layer_idx], src_decoded, trg_decoded):
            if heads_list is None:# 如果没有指定heads_list，就画单个热力图
                self.draw_attention_map(
                    attn_score,
                    cross_attn_score,
                    src,
                    trg,
                )
            else:# 如果指定了heads_list，就画多个热力图
                self.draw_attention_maps(
                    attn_score,
                    cross_attn_score,
                    src,
                    trg,
                    heads_list=heads_list,
                    )
        return [self.mose_detokenizer.tokenize(self.pattern.sub("", s).split()) for s in self.trg_tokenizer.decode(preds)] #将解码后的目标语言句子列表返回，并使用 mose_detokenizer 进行去标记化，最终得到翻译后的结果。


# sentence_list = [
#     "Mann in einem kleinen weißen Boot auf einem See.",  # Man in a small white boat on a lake.
#     "Ein Mann mit einem Eimer und ein Mädchen mit einem Hut am Strand.", # A man with a bucket and a girl in a hat on the beach.
#     "Drei Männer auf Pferden während eines Rennens.",  # Three men on horses during a race.
#     "Ein Mann und eine Frau essen zu Abend",  # 一个男人和一个女人在吃晚餐
# ]
sentence_list = [
    "Mann in einem kleinen weißen Boot auf einem See.",  # Man in a small white boat on a lake.
    "Ein Mann mit einem Eimer und ein Mädchen mit einem Hut am Strand.", # A man with a bucket and a girl in a hat on the beach.
    "Drei Männer auf Pferden während eines Rennens.",  # Three men on horses during a race.
    "Ein Mann und eine Frau essen zu Abend",  # A man and a woman eating dinner
]

# load checkpoints
model = TransformerModel(config)
model.load_state_dict(state_dict)
translator = Translator(model.cpu(), tokenizer, tokenizer)
translator(
    sentence_list,
    layer_idx=-1,
    # heads_list=[0, 1, 2, 3, 4, 5, 6, 7]
    )

```

    Loading vocabulary from ./wmt16/vocab ...
    Read 798250 words (9714 unique) from vocabulary file.
    Loading codes from ./wmt16/bpe.10000 ...
    Read 10001 codes from the codes file.



      0%|          | 0/128 [00:00<?, ?it/s]


    /nasmnt/envs/cosyvoice_vllm_train/lib/python3.10/site-packages/IPython/core/pylabtools.py:170: UserWarning: There are no gridspecs with layoutgrids. Possibly did not call parent GridSpec with the "figure" keyword
      fig.canvas.print_figure(bytes_io, **kw)




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_80_3.png)
    




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_80_4.png)
    




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_80_5.png)
    




![png](transformer_%E5%B8%A6bleu-aliyun_files/transformer_%E5%B8%A6bleu-aliyun_80_6.png)
    





    ['man on a white boat fishing lake with a small boat in the lake.',
     'a man with a backpack and a girl with a backpack on the beach.',
     'three men on a dirt track and the other men are in the [UNK] of them.',
     'a man and woman are enjoying a meal together of food in the same time.']




```python
!ls checkpoints
```

    translate-transformer-not-share



```python
# prompt: 把best.ckpt复制到云盘内

!cp -r checkpoints/translate-transformer-not-share/best.ckpt /content/drive/MyDrive/transformer-de-en

```

    cp: cannot create regular file '/content/drive/MyDrive/transformer-de-en': No such file or directory

