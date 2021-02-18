from numpy.core.fromnumeric import shape
from numpy.core.records import array
from utils import *
from tensorflow.python.keras.layers.core import Dense, Dropout
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import os

def shuffle_arrays_in_unison(arrays):
    fixed_length = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != fixed_length:
            raise Exception('All the arrays need to have same length')
    shuffled_indices = np.random.permutation(fixed_length)
    for i in range(len(arrays)):
        arrays[i] = arrays[i][shuffled_indices]
    return arrays

def load_pickles(src_dir='pose_data'):
    x, y = [],[]
    for fname in os.listdir(src_dir):
        fname = path.join(src_dir, fname)
        data = load_pickle(fname)
        x.extend(data['data'])
        y.extend(data['labels'])
    return np.array(x),np.array(y)

x,y = load_pickles()
x, y = shuffle_arrays_in_unison([x, y])
print('features:', x.shape[-1])
print('total data points:', x.shape[0])
print('POS points:', y.sum())
print('NEG points:', x.shape[0] - y.sum())

split_index = int(x.shape[0]*0.8)
x_train, x_val = x[:split_index], x[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(x.shape[-1],)),
    layers.Dense(8, activation='relu', input_shape=(x.shape[-1],)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer= optimizers.Adam(learning_rate=0.0006),\
    loss='binary_crossentropy',\
    metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=64, epochs=200, validation_data= (x_val, y_val)).history

plt.plot(history['loss'], c='r')
plt.plot(history['val_loss'], c='b')
plt.show()

plt.plot(history['acc'], c='r')
plt.plot(history['val_acc'], c='b')
plt.show()

model.save('saved_models/pose.h5',save_format='h5', include_optimizer=False)

print('done')