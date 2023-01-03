import matplotlib.pyplot as plt
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

