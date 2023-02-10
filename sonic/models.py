from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pathlib

MODEL_PATH = ".models"
model_dir = pathlib.Path(MODEL_PATH)

# now build a class that can be used to export a model and make running
# it later much easier
class ExportModel(tf.Module):
    def __init__(self, model, representation, label_names):
        self.model = model
        self.representation = representation
        self.label_names = label_names

        # Accept either a string-filename or a batch of waveforms.
        # YOu could add additional signatures for a single wave, or a ragged-batch.
        self.__call__.get_concrete_function(x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32)
        )

    @tf.function
    def __call__(self, x):
        # If they pass a string, load the file and decode it.
        if x.dtype == tf.string:
            x = tf.io.read_file(x)
            x, _ = tf.audio.decode_wav(
                x,
                desired_channels=1,
                desired_samples=16000,
            )
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]

        x = self.representation(x)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return {
            "predictions": result,
            "class_ids": class_ids,
            "class_names": class_names,
        }


def sound_model_1(height=173, width=40, channels=1, num_classes=10):
    inputs = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(64, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0004),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, "sound_model_1"


def sound_model_2(height=173, width=40, channels=1, num_classes=10):

    # inputs
    inputs = keras.Input(shape=(height, width, channels))

    # block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # block 4
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Dropout(0.3)(x)

    # block 5
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # outputs
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0004),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, "sound_model_2"


def save_model(model, representation, label_names, model_name="saved"):
    export = ExportModel(model, representation, label_names)
    tf.saved_model.save(export, str(model_dir / model_name))
    return export


def load_model(model_name="saved"):
    imported = tf.saved_model.load(str(model_dir / model_name))
    return imported
