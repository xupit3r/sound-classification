from tensorflow import keras
from tensorflow.keras import layers


def basic_sound_model(height=173, width=40, channels=1, num_classes=10):
  inputs = layers.Input(shape=(height, width, channels))
  flatten = layers.Flatten()(inputs)
  layer1 = layers.Dense(512, activation='relu')(flatten)
  outputs = layers.Dense(num_classes, activation='softmax')(layer1)
  
  model = keras.Model(
    inputs=inputs,
    outputs=outputs
  )

  model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  return model
