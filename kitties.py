import tensorflow as tf
import tensorflow_datasets as tfds


ds = tfds.load("cat_names", split="train", as_supervised=True, shuffle_files=True)

ds = ds.shuffle(1000).batch(128).prefetch(10).take(5)
for audio, label in ds:
    print("label: %s\naudio: %s" % (label, audio))
