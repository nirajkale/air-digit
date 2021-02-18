import tensorflow as tf
import numpy as np

print('loading mnist model..')
model = tf.keras.models.load_model(r'saved_models/mnist.h5')

x = np.zeros((1,28,28,1))
y = model.predict(x)

print('here')