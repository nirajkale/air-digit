from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

ds_train, info = tfds.load('mnist', split='train', with_info=True)
ds_test, info = tfds.load('mnist', split='test', with_info=True)

def preprocess(batch):
    image = batch['image']/255
    label = batch['label']
    return image, label

ds_train = ds_train.map(preprocess).shuffle(buffer_size=50).batch(16).prefetch(AUTOTUNE)
ds_test = ds_test.map(preprocess).shuffle(buffer_size=50).batch(16).prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPool2D(),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPool2D(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPool2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer= optimizers.Adam(learning_rate=0.001),\
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),\
    metrics=['acc'])

history = model.fit(ds_train, epochs=4, validation_data= ds_test).history

plt.plot(history['loss'], c='r')
plt.plot(history['val_loss'], c='b')
plt.show()

plt.plot(history['acc'], c='r')
plt.plot(history['val_acc'], c='b')
plt.show()

model.save('saved_models/mnist.h5',save_format='h5', include_optimizer=False)

print('done')
