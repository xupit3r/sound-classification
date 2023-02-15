from sonic.datasets import kitties
from tensorflow import keras
import numpy as np

train_ds, test_ds, labels = kitties()

train_x = np.array(list(map(lambda v: v[0], train_ds)))
train_y = np.array(list(map(lambda v: v[1], train_ds)))

test_x = np.array(list(map(lambda v: v[0], test_ds)))
test_y = np.array(list(map(lambda v: v[1], test_ds)))

inputs = keras.layers.Input(shape=(82, 129, 1))
x = keras.layers.Dense(256, activation="relu")(inputs)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    train_x, train_y, validation_data=(test_x, test_y), batch_size=20, epochs=20
)
