import librosa
import librosa.display
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


def display_segments(segments):
    for segment in segments:
        plt.plot(segment)
    plt.show()


def display_classes(class_vectors, class_names, x_label="", y_label=""):
    class_idx = 0
    for cv in class_vectors:
        if len(cv.shape) > 1:
            plt.scatter(cv[0, :], cv[1, :], label=class_names[class_idx])
        else:
            plt.scatter(cv[0], cv[1], label=class_names[class_idx])
        class_idx += 1

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def spectrogram(sound_file, n_fft=2048, hop_length=512):
    y, sr = librosa.load(sound_file)
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max
    )

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    # figure 1
    img = librosa.display.specshow(D, y_axis="linear", x_axis="time", sr=sr, ax=ax[0])
    ax[0].set(title="Linear-frequency power spectrogram")
    ax[0].label_outer()

    # figure 2
    librosa.display.specshow(
        D,
        y_axis="log",
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        ax=ax[1],
    )
    ax[1].set(title="Log-frequency power spectrogram")
    ax[1].label_outer()

    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


def mel_spectrogram(sound_file, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(sound_file)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=8000
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    plt.show()
