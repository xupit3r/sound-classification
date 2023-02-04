import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

display_waveforms = False
display_spectrogram = False
display_ds_spectrograms = False


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


# now, let's look at the data set shapes
for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)


# plat some waves for a few classes!
if display_waveforms:
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


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


if display_spectrogram:
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

# cool, now let's convert our training, validation, and test datasets
# to them lovely little spectrograms
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

if display_ds_spectrograms:
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for i in range(n):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(example_spectrograms[i].numpy(), ax)
        ax.set_title(label_names[example_spect_labels[i].numpy()])

    plt.show()

# setup datasets to cache and prefetch to help with performance
train_spectrogram_ds = (
    train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# we will build a CNN to do this work since the spectrograms are 2D
input_shape = example_spectrograms.shape[1:]
print("Input shape:", input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()

# Fit the state of the layer to the spectrograms with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential(
    [
        layers.Input(shape=input_shape),
        # Downsample the input (for quicker training)
        # layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        # push through the CNN
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ]
)

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


# plot training and validation loss curves
metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
plt.legend(["loss", "val_loss"])
plt.ylim([0, max(plt.ylim())])
plt.xlabel("Epoch")
plt.ylabel("Loss [CrossEntropy]")

plt.subplot(1, 2, 2)
plt.plot(
    history.epoch,
    100 * np.array(metrics["accuracy"]),
    100 * np.array(metrics["val_accuracy"]),
)
plt.legend(["accuracy", "val_accuracy"])
plt.ylim([0, 100])
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.show()

# evaluate the model against our test data
model.evaluate(test_spectrogram_ds, return_dict=True)

# display a confusion matrix to visualize the results
y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt="g"
)
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.show()
