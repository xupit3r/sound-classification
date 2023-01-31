from sonic.features import extract_features, extract_directories
from sonic.interesting import interesting_segments
from sonic.preprocess import denoise, signal_frames
from sonic.review import (
    display_classes,
    mel_spectrogram,
    spectral_bandwidth,
    spectral_centroid,
    spectral_rolloff,
    spectrogram,
)
from sonic.utils import get_sound_dirs
import matplotlib.pyplot as plt
import soundfile as sf

SOUND_FILES = ".test_data"


def show_interesting_segments(SOUND_FILES):
    sound_file = f"{SOUND_FILES}/city/traffic/city-traffic-outdoor.wav"

    vector = extract_features(sound_file=sound_file)

    print(vector.shape)

    TEST_DIRECTORIES = get_sound_dirs(f"{SOUND_FILES}/city")

    class_vectors, class_names = extract_directories(dirs=TEST_DIRECTORIES)

    display_classes(
        class_vectors, class_names, "spectral_centroid_mean", "energy_entropy_mean"
    )

    interesting, blocks, signal = interesting_segments(
        f"{SOUND_FILES}/city/ambience/street-ambience.wav", base_block=1, base_step=0.05
    )

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

    # plot the original signal
    ax[0].set_title("original")
    ax[0].plot(signal)

    # plot the blocks view of the interesting segments
    ax[1].set_title("blocks")
    ax[1].plot(blocks)

    # plot the intersting signal (combined blocks)
    ax[2].set_title("interesting")
    ax[2].plot(interesting)

    plt.show()


def show_denoising():
    signal, sample_rate = sf.read(f"{SOUND_FILES}/city/ambience/street-ambience.wav")

    denoised = denoise(signal)

    fig, axs = plt.subplots(2)
    axs[0].set_title("original signal")
    axs[0].plot(signal)
    axs[1].set_title("denoised signal")
    axs[1].plot(denoised)
    plt.show()

    # # prepare frames from the denoised signal
    frames = signal_frames(denoised, sample_rate)

    plt.plot(frames)
    plt.show()


def show_spectrogram():
    spectrogram(f"{SOUND_FILES}/anomaly/joe.wav")


def show_mel_spectrogram():
    mel_spectrogram(f"{SOUND_FILES}/anomaly/joe.wav")


def show_spectral_centroid():
    spectral_centroid(f"{SOUND_FILES}/anomaly/joe.wav")


def show_spectral_rolloff():
    spectral_rolloff(f"{SOUND_FILES}/anomaly/joe.wav")


def show_spectral_bandwidth():
    spectral_bandwidth(f"{SOUND_FILES}/anomaly/joe.wav")
