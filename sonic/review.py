import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# plot the result of a training run
def plot_result(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, "bo", label="Training loss")
  plt.plot(epochs, val_loss, "b", label="Validation loss")
  plt.title("Training and validation loss")
  plt.legend()
  plt.show()

def display_audio(sound_file):
  sound, sample_rate = sf.read(sound_file)
  signal = np.array(sound)
  plt.plot(signal)
  plt.show()