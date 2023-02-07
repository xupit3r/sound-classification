from sonic.features import mfcc
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
