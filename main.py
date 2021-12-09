import os
import random
import csv
import numpy as np
from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Pad, Stack, Tuple

from utils import load_vocab, convert_example

#分别为训练数据、测试数据的文件路径
original_data = 'challenging/data/train.csv'
test_data = 'challenging/data/test.csv'

#谣言标签为0，非谣言标签为1
rumor_label = "1"
non_rumor_label = "0"

#分别统计谣言数据与非谣言数据的总数
rumor_num = 0
non_rumor_num = 0
test_num = 0

all_rumor_list = []
all_non_rumor_list = []
test_list = []

#解析训练数据
with open(original_data, "r", encoding='utf-8') as f:
    text = csv.reader(f)
    for i in text:
        if i[5] != "1" and i[5] != "0":
            continue
        if i[5] == "1":
            all_rumor_list.append(i[0] + i[1] + "\t" + i[5] + "\n")
            rumor_num += 1
        else:
            all_non_rumor_list.append(i[0] + i[1] + "\t" + i[5] + "\n")
            non_rumor_num += 1

with open(test_data, "r", encoding='utf-8') as f:
    text = csv.reader(f)
    for i in text:
        if i[5] != "1" and i[5] != "0":
            continue
        test_list.append(i[0] + i[1] + "\t" + i[5] + "\n")
        test_num += 1

print("谣言数据总量为：" + str(rumor_num))
print("非谣言数据总量为：" + str(non_rumor_num))
data_list_path = "data/"
all_data_path = data_list_path + "all_data.txt"
test_data_path = data_list_path + "test_data.txt"
all_data_list = all_rumor_list + all_non_rumor_list

random.shuffle(all_data_list)
random.shuffle(test_list)

#在生成all_data.txt之前，首先将其清空
with open(all_data_path, 'w') as f:
    f.seek(0)
    f.truncate()

with open(test_data_path, 'w') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a') as f:
    for data in all_data_list:
        f.write(data)

with open(test_data_path, 'a') as f:
    for data in test_list:
        f.write(data)

with open(os.path.join(data_list_path, 'eval_list.txt'), 'w',
          encoding='utf-8') as f_eval:
    f_eval.seek(0)
    f_eval.truncate()

with open(os.path.join(data_list_path, 'train_list.txt'),
          'w',
          encoding='utf-8') as f_train:
    f_train.seek(0)
    f_train.truncate()

with open(os.path.join(data_list_path, 'all_data.txt'), 'r',
          encoding='utf-8') as f_data:
    lines = f_data.readlines()

with open(os.path.join(data_list_path, 'test_data.txt'), 'r',
          encoding='utf-8') as f_test_init:
    eval_lines = f_test_init.readlines()

i = 0
with open(os.path.join(data_list_path, 'train_list.txt'),
          'a',
          encoding='utf-8') as f_train:
    for line in lines:
        words = line.split('\t')[-1].replace('\n', '')
        label = line.split('\t')[0]
        labs = ""
        labs = label + '\t' + words + '\n'
        f_train.write(labs)
        i += 1

j = 0
with open(os.path.join(data_list_path, 'eval_list.txt'), 'a',
          encoding='utf-8') as f_test:
    for line in eval_lines:
        words = line.split('\t')[-1].replace('\n', '')
        label = line.split('\t')[0]
        labs = ""
        labs = label + '\t' + words + '\n'
        f_test.write(labs)
        j += 1

print("数据列表生成完成！")


class SelfDefinedDataset(paddle.io.Dataset):
    def __init__(self, data):
        super(SelfDefinedDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return ["0", "1"]


def txt_to_list(file_name):
    res_list = []
    for line in open(file_name):
        res_list.append(line.strip().split('\t'))
    return res_list


trainlst = txt_to_list('data/train_list.txt')
devlst = txt_to_list('data/eval_list.txt')
# testlst = txt_to_list('test.txt')

# 通过get_datasets()函数，将list数据转换为dataset。
# get_datasets()可接收[list]参数，或[str]参数，根据自定义数据集的写法自由选择。
# train_ds, dev_ds, test_ds = ppnlp.datasets.ChnSentiCorp.get_datasets(['train', 'dev', 'test'])
train_ds, dev_ds = SelfDefinedDataset.get_datasets([trainlst, devlst])
label_list = train_ds.get_labels()
print(label_list)

for i in range(10):
    print(train_ds[i])

import jieba

dict_path = 'data/dict.txt'

#创建数据字典，存放位置：dicts.txt。在生成之前先清空dict.txt
#在生成all_data.txt之前，首先将其清空
with open(dict_path, 'w') as f:
    f.seek(0)
    f.truncate()

dict_set = set()
train_data = open('data/train_list.txt')
for data in train_data:
    seg = jieba.lcut(data[:-3])
    for datas in seg:
        if not datas is " ":
            dict_set.add(datas)

dicts = open(dict_path, 'w')
dicts.write('[PAD]\n')
dicts.write('[UNK]\n')
for data in dict_set:
    dicts.write(data + '\n')
dicts.close()
vocab = load_vocab(dict_path)

for k, v in vocab.items():
    print(k, v)
    break


# Reads data and generates mini-batches.
def create_dataloader(dataset,
                      trans_function=None,
                      mode='train',
                      batch_size=1,
                      pad_token_id=0,
                      batchify_fn=None):
    if trans_function:
        dataset = dataset.apply(trans_function, lazy=True)

    # return_list 数据是否以list形式返回
    # collate_fn  指定如何将样本列表组合为mini-batch数据。传给它参数需要是一个callable对象，需要实现对组建的batch的处理逻辑，并返回每个batch的数据。在这里传入的是`prepare_input`函数，对产生的数据进行pad操作，并返回实际长度等。
    dataloader = paddle.io.DataLoader(dataset,
                                      return_list=True,
                                      batch_size=batch_size,
                                      collate_fn=batchify_fn)

    return dataloader


# python中的偏函数partial，把一个函数的某些参数固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。
trans_function = partial(convert_example,
                         vocab=vocab,
                         unk_token_id=vocab.get('[UNK]', 1),
                         is_test=False)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab['[PAD]']),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]

train_loader = create_dataloader(train_ds,
                                 trans_function=trans_function,
                                 batch_size=32,
                                 mode='train',
                                 batchify_fn=batchify_fn)
dev_loader = create_dataloader(dev_ds,
                               trans_function=trans_function,
                               batch_size=32,
                               mode='validation',
                               batchify_fn=batchify_fn)


class LSTMModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)

        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)

        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)

        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # text shape: (batch_size, num_tokens)
        # print('input :', text.shape)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # print('after word-embeding:', embedded_text.shape)

        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirectional' else 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # print('after lstm:', text_repr.shape)

        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # print('after Linear classifier:', fc_out.shape)

        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # print('output:', logits.shape)

        # probs 分类概率值
        probs = F.softmax(logits, axis=-1)
        # print('output probability:', probs.shape)
        return probs


model = LSTMModel(len(vocab),
                  len(label_list),
                  direction='bidirectional',
                  padding_idx=vocab['[PAD]'])
model = paddle.Model(model)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                  learning_rate=5e-5)

loss = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

model.prepare(optimizer, loss, metric)
# 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
model.fit(train_loader,
          dev_loader,
          epochs=10,
          save_dir='./checkpoints',
          save_freq=5,
          callbacks=callback)
results = model.evaluate(dev_loader)
print("Finally test acc: %.5f" % results['acc'])
label_map = {1: '谣言', 0: '非谣言'}
results = model.predict(dev_loader, batch_size=128)[0]
predictions = []

for batch_probs in results:
    # 映射分类label
    idx = np.argmax(batch_probs, axis=-1)
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    predictions.extend(labels)

# 看看预测数据前5个样例分类结果
for idx, data in enumerate(dev_ds.data[:10]):
    print('Data: {} \t Label: {}'.format(data[0], predictions[idx]))