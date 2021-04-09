# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:28:14 2021

@author: qg
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
#使用 Pandas 创建一个 dataframe

dataframe = pd.read_csv('D:/tensorflow_test/heart/heart.csv')
dataframe.head()
dataframe = (dataframe - np.min(dataframe)) / (np.max(dataframe) - np.min(dataframe))
#print(dataframe.head(5))
#将数据拆分为训练、验证和测试

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


#使用tf.data创建输入管道
# 一种从Pandas Dataframe创建tf.data数据集的使用方法 
 
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()#原代码dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 5 # 小批量用于演示目的
train_ds = df_to_dataset(train, shuffle=False,batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
#现在我们已经创建了输入管道，让我们调用它来查看它返回的数据的格式，我们使用了一小批量来保持输出的可读性。
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch)

# 我们将使用此批处理来演示几种类型的特征列 
example_batch = next(iter(train_ds))[0]

# 用于创建特征列和转换批量数据 
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())


feature_columns = []

# numeric 数字列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))
age = feature_column.numeric_column("age")
# bucketized 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)


# indicator 指示符列 
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding 嵌入列 
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed 交叉列 
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
#创建特征层
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#创建了一个具有更大批量的新输入管道
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
#创建、编译和训练模型
model = keras.Sequential([
  feature_layer,
  layers.Dense(128, activation=tf.nn.relu),
  layers.Dense(128, activation=tf.nn.relu),
  layers.Dense(1, activation=tf.nn.softmax)
])

model.compile(tf.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,val_ds,epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


