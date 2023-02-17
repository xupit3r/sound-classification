import os

from librosa import resample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

testing_wav_file_name = tf.keras.utils.get_file(
    "miaow_16k.wav",
    "https://storage.googleapis.com/audioset/miaow_16k.wav",
    cache_dir="./",
    cache_subdir=".test_data",
)

# Utility functions for loading audio files and making sure the sample rate is correct.
@tf.function
def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio."""
    file_contents = tf.io.read_file(filename)
    signal, sr = tf.audio.decode_wav(
        file_contents, desired_channels=1, desired_samples=16000
    )
    signal = tf.squeeze(signal, axis=-1)
    sr = tf.cast(sr, dtype=tf.int64)
    return signal


testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

# plt.plot(testing_wav_data)
# plt.show()

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
class_names = list(pd.read_csv(class_map_path)["display_name"])

for name in class_names[:20]:
    print(name)
print("...")

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]

print(f"The main sound is: {inferred_class}")
print(f"The embeddings shape: {embeddings.shape}")

_ = tf.keras.utils.get_file(
    "esc-50.zip",
    "https://github.com/karoldvl/ESC-50/archive/master.zip",
    cache_dir=".datasets",
    cache_subdir="tensorflow",
    extract=True,
)

esc50_csv = ".datasets/tensorflow/ESC-50-master/meta/esc50.csv"
base_data_path = ".datasets/tensorflow/ESC-50-master/audio/"

pd_data = pd.read_csv(esc50_csv)
print(pd_data.head())

my_classes = ["dog", "cat"]
map_class_to_id = {"dog": 0, "cat": 1}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd["category"].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd["filename"].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

print(filtered_pd.head(10))

filenames = filtered_pd["filename"]
targets = filtered_pd["target"]
folds = filtered_pd["fold"]

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


main_ds = main_ds.map(load_wav_for_map)
print(main_ds.element_spec)


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
    """run YAMNet to extract embedding from the wav data"""
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (
        embeddings,
        tf.repeat(label, num_embeddings),
        tf.repeat(fold, num_embeddings),
    )


# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
print(main_ds.element_spec)

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
