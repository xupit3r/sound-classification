from sonic.datasets import build_urban_sounds, get_urban_sounds
from sonic.gpu import setup_gpus
from sonic.models import basic_sound_model

height = 173
width = 40
channels = 1

setup_gpus()

# build_urban_sounds()
model = basic_sound_model(
  height=height,
  width=width,
  channels=channels
)
ds_train, ds_val, class_names = get_urban_sounds()

model.summary()

history = model.fit(
  ds_train[0],
  ds_train[1],
  validation_data=ds_val,
  epochs=20,
  callbacks=[],
  batch_size=8
)