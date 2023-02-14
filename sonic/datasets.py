from functools import reduce
from re import sub
from sonic.features import mfcc
from sonic.review import get_spectrogram, plot_spectrogram
from sklearn.model_selection import train_test_split
from operator import itemgetter
from tqdm import tqdm
import os
import pathlib
import csv as csv
import librosa as librosa
import cv2 as cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import soundfile as sf

DATASETS_DIR = ".datasets"
BINARY_OUTPUT = ".cached_datasets"
TENSORFLOW_DATASETS = f"{DATASETS_DIR}/tensorflow"

pathlib.Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(BINARY_OUTPUT).mkdir(parents=True, exist_ok=True)
pathlib.Path(TENSORFLOW_DATASETS).mkdir(parents=True, exist_ok=True)

# ensure audio is only dealing with one channel
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_tensorflow_dataset():
    data_dir = pathlib.Path(f"{TENSORFLOW_DATASETS}/mini_speech_commands")

    if not data_dir.exists():
        tf.keras.utils.get_file(
            "mini_speech_commands.zip",
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir=TENSORFLOW_DATASETS,
            cache_subdir=".",
        )

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

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # keep a separate test dataset using shards!
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    return train_ds, val_ds, test_ds, label_names, data_dir


def get_dataset_examples(train_ds):
    for example_audio, example_labels in train_ds.take(1):
        break
    return example_audio, example_labels


def plot_example_waveforms(label_names, example_audio, example_labels):
    label_names[[1, 1, 3, 0]]

    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for i in range(n):
        if i >= n:
            break
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(example_audio[i].numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label_names[example_labels[i]]
        ax.set_title(label)
        ax.set_ylim([-1.1, 1.1])

    plt.show()


def plot_example_spectrogram(label_names, example_audio, example_labels):
    for i in range(3):
        label = label_names[example_labels[i]]
        waveform = example_audio[i]
        spectrogram = get_spectrogram(waveform)

        print("Label:", label)
        print("Waveform shape:", waveform.shape)
        print("Spectrogram shape:", spectrogram.shape)
        print("Audio playback")

        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(waveform.shape[0])
        axes[0].plot(timescale, waveform.numpy())
        axes[0].set_title("Waveform")
        axes[0].set_xlim([0, 16000])

        plot_spectrogram(spectrogram.numpy(), axes[1])
        axes[1].set_title("Spectrogram")
        plt.suptitle(label.title())
        plt.show()


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def get_spectrogram_examples(train_spectrogram_ds):
    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break
    return example_spectrograms, example_spect_labels


def read_dataset_info(name="", file_key="", class_key="", classname_key=""):
    with open(f"{DATASETS_DIR}/{name}/config.csv") as file:
        reader = csv.DictReader(file, delimiter=",")

        info = []
        for row in reader:
            info.append([row[file_key], int(row[class_key]), row[classname_key]])

        class_names = get_classnames(info)
        num_classes = len(class_names)

        return info, num_classes, class_names


def read_wav_files(dataset_name="", dir="", height=173, width=40, feature=mfcc):
    if len(dir) == 0:
        dir = "sounds"

    files = os.listdir(f"{DATASETS_DIR}/{dataset_name}/{dir}")

    sound_files = {}
    for file in tqdm(files):
        raw, sample_rate = librosa.load(
            f"{DATASETS_DIR}/{dataset_name}/{dir}/{file}", res_type="kaiser_fast"
        )
        audio = feature(raw, sample_rate)
        sound_files[file] = cv2.resize(
            audio, (width, height), interpolation=cv2.INTER_LINEAR
        )

    return sound_files


def get_classnames(info):
    return list(set(map(itemgetter(-1), info)))


def to_one_hot(label, dimension=10):
    one_hot = np.zeros((dimension,))
    one_hot[label] = 1.0
    return one_hot


def build_urban_sounds(height=173, width=40, feature=mfcc):
    info, num_classes, class_names = read_dataset_info(
        name="urban_sounds",
        file_key="slice_file_name",
        class_key="classID",
        classname_key="class",
    )

    print("processing sound files")
    sound_files = read_wav_files(
        dataset_name="urban_sounds",
        dir="sounds",
        height=height,
        width=width,
        feature=feature,
    )

    x = []
    y = []
    for pair in info:
        x.append(sound_files[pair[0]])
        y.append(to_one_hot(pair[1], num_classes))

    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    y = np.array(y)

    print(f"x shape {x.shape}")
    print(f"y shape {y.shape}")

    # save these out to the filesystem for reuse
    np.savez(f"{BINARY_OUTPUT}/urban_sounds.npz", x, y)


def get_urban_sounds(feature=mfcc):
    info, num_classes, class_names = read_dataset_info(
        name="urban_sounds",
        file_key="slice_file_name",
        class_key="classID",
        classname_key="class",
    )

    npz = np.load(f"{BINARY_OUTPUT}/urban_sounds.npz")
    x = npz["arr_0"]
    y = npz["arr_1"]
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return ((train_x, train_y), (test_x, test_y), class_names)


def kitties():
    ds = {"train": [], "test": []}

    def createEntry(subdir, file):
        key = "train" if subdir.endswith("train_data") else "test"
        signal, sample_rate = sf.read(os.path.join(subdir, file))
        label = to_one_hot(1 if file.startswith("ada") else 0, 2)
        ds[key].append((get_spectrogram(signal), label))

    for subdir, dirs, files in os.walk("sonic/custom_datasets/cat_names"):
        for file in files:
            if file.endswith(".wav"):
                createEntry(subdir, file)

    return ds
