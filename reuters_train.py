#!/usr/bin/env python3

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import numpy as np
from reuters_model import createHierarchicalAttentionModel

batch_size = 16
max_features = 20000
maxlen = 80 # 为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0

print('loading data...')
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz", num_words=max_features, test_split=0.2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(len(X_train), type(X_train), X_train.shape)
print(len(X_test), type(X_test), X_test.shape)
print('wds')
print(y_test[0:300])
#8982 <class 'numpy.ndarray'> (8982,)
#2246 <class 'numpy.ndarray'> (2246,)
print(len(X_train[0]), X_train[0], '\n')
print(len(X_train[1]), X_train[1], '\n')

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print(len(X_train), 'train sequences', X_train.shape)
print(len(X_test), 'test sequences', X_test.shape)
#8982 train sequences (8982, 80)
#2246 test sequences (2246, 80)
print(len(X_train[0]), X_train[0], '\n')
print(len(X_train[1]), X_train[1], '\n')

# add one extra dimention as the sentence (1 sentence per doc!)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
#X_train shape: (8982, 1, 80)
#X_test shape: (2246, 1, 80)

print('Build model...')
model = createHierarchicalAttentionModel(maxlen, embeddingSize=200, vocabSize=max_features)

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, epochs=15, verbose=1,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

print('end')
