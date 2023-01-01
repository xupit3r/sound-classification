import tensorflow as tf

# sets up the GPU config
def setup_gpus(use_all=False):
  if not use_all:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # turn memory growth on for each of my gpus (well, the one...)
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "physical GPUs,", len(logical_gpus), "logical GPUs")
      except RuntimeError as e:
        # was it set before hand?
        print(e)