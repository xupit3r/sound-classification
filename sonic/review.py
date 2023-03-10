import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import tensorflow as tf
import seaborn as sns

# plot the result of a training run
def plot_result(history):
    metrics = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch")
    plt.ylabel("Loss [CrossEntropy]")

    plt.subplot(1, 2, 2)
    plt.plot(
        history.epoch,
        100 * np.array(metrics["accuracy"]),
        100 * np.array(metrics["val_accuracy"]),
    )
    plt.legend(["accuracy", "val_accuracy"])
    plt.ylim([0, 100])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.show()


def confusion_maxtrix(model, test_spectrogram_ds, label_names):
    # prepare confusion matrix data
    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

    # plot the confusion matrix
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mtx,
        xticklabels=label_names,
        yticklabels=label_names,
        annot=True,
        fmt="g",
    )
    plt.xlabel("Prediction")
    plt.ylabel("Label")


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


def spectral_centroid(sound_file):
    y, sr = librosa.load(sound_file)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    S, phase = librosa.magphase(librosa.stft(y=y))
    times = librosa.times_like(cent)
    fig, ax = plt.subplots()
    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax
    )
    ax.plot(times, cent.T, label="Spectral centroid", color="w")
    ax.legend(loc="upper right")
    ax.set(title="log Power spectrogram")
    plt.show()


def spectral_rolloff(sound_file):
    y, sr = librosa.load(sound_file)

    # an approximation of the max rolloff
    max_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)

    # an approximation of the min rolloff
    min_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)

    # separate out the maginitude (S) and phase components
    S, phase = librosa.magphase(librosa.stft(y))

    # plot!
    fig, ax = plt.subplots()
    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax
    )

    # plot the max rolloff line
    ax.plot(
        librosa.times_like(max_rolloff),
        max_rolloff[0],
        label="Roll-off frequency (0.99)",
    )

    # plot the min rolloff line
    ax.plot(
        librosa.times_like(max_rolloff),
        min_rolloff[0],
        color="w",
        label="Roll-off frequency (0.01)",
    )

    ax.legend(loc="lower right")
    ax.set(title="log Power spectrogram")
    plt.show()


def spectral_bandwidth(sound_file):
    y, sr = librosa.load(sound_file)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    S, phase = librosa.magphase(librosa.stft(y=y))

    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(spec_bw)
    centroid = librosa.feature.spectral_centroid(S=S)

    ax[0].semilogy(times, spec_bw[0], label="Spectral bandwidth")
    ax[0].set(ylabel="Hz", xticks=[], xlim=[times.min(), times.max()])
    ax[0].legend()
    ax[0].label_outer()

    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax[1]
    )

    ax[1].set(title="log Power spectrogram")
    ax[1].fill_between(
        times,
        np.maximum(0, centroid[0] - spec_bw[0]),
        np.minimum(centroid[0] + spec_bw[0], sr / 2),
        alpha=0.5,
        label="Centroid +- bandwidth",
    )
    ax[1].plot(times, centroid[0], label="Spectral centroid", color="w")
    ax[1].legend(loc="lower right")
    plt.show()


def spectral_contrast(sound_file):
    y, sr = librosa.load(sound_file)
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    img1 = librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax[0]
    )

    fig.colorbar(img1, ax=[ax[0]], format="%+2.0f dB")
    ax[0].set(title="Power spectrogram")
    ax[0].label_outer()

    img2 = librosa.display.specshow(contrast, x_axis="time", ax=ax[1])
    fig.colorbar(img2, ax=[ax[1]])
    ax[1].set(ylabel="Frequency bands", title="Spectral contrast")

    plt.show()


def tonnetz(sound_file):
    y, sr = librosa.load(sound_file)
    y = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    img1 = librosa.display.specshow(tonnetz, y_axis="tonnetz", x_axis="time", ax=ax[0])
    ax[0].set(title="Tonal Centroids (Tonnetz)")
    ax[0].label_outer()

    img2 = librosa.display.specshow(
        librosa.feature.chroma_cqt(y=y, sr=sr), y_axis="chroma", x_axis="time", ax=ax[1]
    )
    ax[1].set(title="Chroma")
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])

    plt.show()


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
