import os

from librosa import resample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf

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
    signal, sr = sf.read(filename)
    signal = resample(signal, orig_sr=sr, target_sr=16000)

    # yamnet expects tf.float32 as input...
    signal = tf.cast(signal, dtype=tf.float32)
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
