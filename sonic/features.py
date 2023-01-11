from pyAudioAnalysis import ShortTermFeatures as af
import soundfile as sf
import numpy as np

def extract_features(sound_file='', window=0.050, step=0.050, features=['mfcc_1', 'spectral_centroid']):
  signal, sample_rate = sf.read(sound_file)
  # duration = len(signal) / float(sample_rate)

  # extract short-term features
  [feature_values, feature_names] = af.feature_extraction(
    signal, 
    sample_rate, 
    int(sample_rate * window), 
    int(sample_rate * step)
  )

  extracted = list(
    map(
      lambda feature: feature_values[feature_names.index(feature), :], 
      features
    )
  )

  return np.array(extracted)
