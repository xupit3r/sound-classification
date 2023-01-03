from tensorflow import keras
from tensorflow.keras import layers, regularizers


def sound_model_1(height=173, width=40, channels=1, num_classes=10):
  inputs = layers.Input(shape=(height, width, channels))
  x = layers.Conv2D(64, 3, activation='relu')(inputs)
  x = layers.MaxPooling2D(2, padding='same')(x)
  x = layers.Conv2D(256, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2, padding='same')(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Conv2D(256, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2, padding='same')(x)
  x = layers.Dropout(0.3)(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(256, activation='relu')(x)
  outputs = layers.Dense(num_classes, activation='softmax')(x)
  
  model = keras.Model(
    inputs=inputs,
    outputs=outputs
  )

  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0004),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  return model, 'sound_model_1'

def sound_model_2(height=173, width=40, channels=1, num_classes=10):

  # inputs
  inputs = keras.Input(shape=(height, width, channels))

  # block 2
  x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
  x = layers.BatchNormalization()(x)

  # block 3
  x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  # block 4
  x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
  x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
  x = layers.MaxPooling2D(2, padding='same')(x)
  x = layers.Dropout(0.3)(x)

  # block 5
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(
    256,
    activation='relu'
  )(x)
  x = layers.Dropout(0.5)(x)

  # outputs
  outputs = layers.Dense(num_classes, activation='softmax')(x)

  model = keras.Model(
    inputs=inputs,
    outputs=outputs
  )

  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0004),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  return model, 'sound_model_2'