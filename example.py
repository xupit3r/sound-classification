from sonic.datasets import (
    get_spectrogram_examples,
    get_tensorflow_dataset,
    get_dataset_examples,
    plot_example_spectrogram,
    plot_example_waveforms,
    make_spec_ds,
)
from sonic.review import get_spectrogram
from sonic.models import save_model, load_model
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

# NOTE: this example follows: https://www.tensorflow.org/tutorials/audio/simple_audio

display_waveforms = False
display_spectrogram = False
display_ds_spectrograms = False


# good idea to set random seed to be able to reproduce
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

MODEL_PATH = ".models"
model_dir = pathlib.Path(MODEL_PATH)

train_ds, val_ds, test_ds, label_names, data_dir = get_tensorflow_dataset()

# now, let's look at the data set shapes
example_audio, example_labels = get_dataset_examples(train_ds)

# plat some waves for a few classes!
if display_waveforms:
    plot_example_waveforms(label_names, example_audio, example_labels)


if display_spectrogram:
    plot_example_spectrogram(label_names, example_audio, example_labels)

# cool, now let's convert our training, validation, and test datasets
# to them lovely little spectrograms
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

# get the first batch of example spectrograms
example_spectrograms, example_spect_labels = get_spectrogram_examples(
    train_spectrogram_ds
)

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
        layers.Resizing(32, 32),
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


# finally, let's run an inference on an audio file that says "no"
x = data_dir / "no/01bb6a2a_nohash_0.wav"
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(
    x,
    desired_channels=1,
    desired_samples=16000,
)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis, ...]

prediction = model(x)
x_labels = ["no", "yes", "down", "go", "left", "up", "right", "stop"]
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title("No")
plt.show()

# save the model and reload it
save_model(model, get_spectrogram, label_names, "example_model")
imported = load_model("example_model")

# test the loaded model
print(imported(waveform[tf.newaxis, :]))
