import tensorflow as tf
from tensorflow.python.client import device_lib
from sonic.gpu import setup_gpus

tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)

setup_gpus()

print(tf.__version__)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
