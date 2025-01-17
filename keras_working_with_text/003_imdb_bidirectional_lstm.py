#!/usr/bin/python
# -*- coding: utf-8 -*-

# Train a Bidirectional LSTM on the IMDB sentiment classification task.
# 利用Bi-LSTM模型对IMDB影评倾向分类

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

max_features = 20000    # vocab大小
batch_size = 32         # min-batch size
maxlen = 100            # 每条样本数据长度

# 数据集来源IMDB影评,共50000条影评,标记正面/负面两种评价
# 每条数据被编码为一条索引序列(索引数字越小,代表单词出现次数越多)
# num_words: 选取的每条数据里的索引值不能超过num_words
print('========== 1.Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print('----- train sequences', len(x_train))
print('----- test  sequences', len(x_test))

# 对每条词索引组成的数据进行长度对齐,去掉数据前面或后面多余的单词;长度不够插入0
print('========== 2.Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('----- x_train shape:', x_train.shape)
print('----- x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 搭建神经网络模型
print('========== 3.Build model...')
model = Sequential()
# input_dim=max_features单词表大小,output_dim=128为词向量维度,input_length=maxlen每条样本数据长度
model.add(Embedding(max_features, 128, input_length=maxlen))    # 将正整数下标转换为具有固定大小的向量,输出(*,100,128)
# units=64代表通过LSTM,词向量维度转换为64
model.add(Bidirectional(LSTM(64)))    # 输出(*, 128)
model.add(Dropout(0.5))               # 输出(*, 128)
model.add(Dense(1, activation='sigmoid'))

# 神经网络编译/训练/测试集测试性能
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=[x_test, y_test])
