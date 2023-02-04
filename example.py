import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

# good idea to set random seed to be able to reproduce
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = ".cached_datasets/tensorflow/mini_speech_commands"
data_dir = pathlib.Path(DATASET_PATH)

if not data_dir.exists():
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=".cached_datasets",
        cache_subdir="tensorflow",
    )

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != "README.md") & (commands != ".DS_Store")]
print("Commands:", commands)

# build the training/validation datasets
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset="both",
)

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

# ensure audio is only dealing with one channel
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# keep a separate test dataset using shards!
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)
