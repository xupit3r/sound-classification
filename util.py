import tensorflow as tf
import tensorflow_datasets as tfds
import soundfile as sf
from sonic.gpu import setup_gpus

tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)

setup_gpus()

print(tf.__version__)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

signal, sample_rate = sf.read("sonic/datasets/cat_names/train_data/ada_20.wav")
print(sample_rate)

ds = tfds.load("cat_names")
