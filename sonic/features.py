import os
import numpy as np
import soundfile as sf
from pyAudioAnalysis import MidTermFeatures as af

CHUNKS = {"mid_window": 1, "mid_step": 1, "short_window": 0.1, "short_step": 0.05}

FEATURES = ["spectral_centroid_mean", "energy_entropy_mean"]


def get_energies(segments=[]):
    return [(s**2).sum() / len(s) for s in segments]


def pull_specified(feature_values, feature_names, to_extract):
    if len(feature_values.shape) > 1:
        extracted = list(
            map(
                lambda feature: feature_values[:, feature_names.index(feature)],
                to_extract,
            )
        )
    else:
        extracted = list(
            map(
                lambda feature: feature_values[feature_names.index(feature)], to_extract
            )
        )

    return np.array(extracted)


def extract_features(sound_file="", chunks=CHUNKS, features=FEATURES):
    signal, sample_rate = sf.read(sound_file)

    # extract short-term features
    mid_features, short_features, feature_names = af.mid_feature_extraction(
        signal,
        sample_rate,
        int(sample_rate * chunks["mid_window"]),
        int(sample_rate * chunks["mid_step"]),
        int(sample_rate * chunks["short_window"]),
        int(sample_rate * chunks["short_step"]),
    )

    print(f"{short_features.shape[1]} {short_features.shape[0]} short term vectors")
    print(f"{mid_features.shape[1]} {mid_features.shape[0]} mid term vectors")

    extracted = list(
        map(lambda feature: mid_features[feature_names.index(feature), :], features)
    )

    return np.array(extracted)


def extract_directories(dirs=[], chunks=CHUNKS, features_to_extract=FEATURES):
    class_names = [os.path.basename(d) for d in dirs]

    # pull features for each directory (i.e. class)
    features = []
    for d in dirs:
        mid_features, files, feature_names = af.directory_feature_extraction(
            d,
            chunks["mid_window"],
            chunks["mid_step"],
            chunks["short_window"],
            chunks["short_step"],
        )
        features.append(mid_features)

    # create a feature maxtrix for each of the classes
    extracted = list(
        map(
            lambda feature: pull_specified(feature, feature_names, features_to_extract),
            features,
        )
    )

    return extracted, class_names
