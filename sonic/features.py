from pyAudioAnalysis import MidTermFeatures as mid_af
import soundfile as sf
import numpy as np

CHUNKS = {
  'mid_window': 1,
  'mid_step': 1,
  'short_window': 0.050,
  'short_step': 0.050
}

FEATURES = [
  'spectral_centroid_mean',
  'energy_entropy_mean'
]

def extract_features(sound_file='', chunks=CHUNKS, features=FEATURES):
  signal, sample_rate = sf.read(sound_file)

  # extract short-term features
  mid_features, short_features, feature_names = mid_af.mid_feature_extraction(
    signal, 
    sample_rate, 
    int(sample_rate * chunks['mid_window']), 
    int(sample_rate * chunks['mid_step']),
    int(sample_rate * chunks['short_window']), 
    int(sample_rate * chunks['short_step'])
  )

  print(f'{short_features.shape[1]} {short_features.shape[0]} short term vectors')
  print(f'{mid_features.shape[1]} {mid_features.shape[0]} mid term vectors')

  extracted = list(
    map(
      lambda feature: mid_features[feature_names.index(feature), :], 
      features
    )
  )

  return np.array(extracted)