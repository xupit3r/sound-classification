from sonic.features import extract_features, extract_directories
from sonic.interesting import interesting_segments
from sonic.preprocess import denoise
from sonic.review import display_classes, display_audio, display_segments
from sonic.utils import get_sound_dirs
import matplotlib.pyplot as plt
import os

SOUND_FILES = '.test_data'
SEGMENTS_DIR = f'{SOUND_FILES}/segments'

if not os.path.exists(SEGMENTS_DIR):
  os.makedirs(SEGMENTS_DIR)

# sound_file = f'{SOUND_FILES}/city/traffic/city-traffic-outdoor.wav'

# vector = extract_features(
#   sound_file=sound_file
# )

# print(vector.shape)

# TEST_DIRECTORIES = get_sound_dirs(f'{SOUND_FILES}/city')

# class_vectors, class_names = extract_directories(dirs=TEST_DIRECTORIES)

# display_classes(
#   class_vectors,
#   class_names,
#   'spectral_centroid_mean',
#   'energy_entropy_mean'
# )

interesting, blocks, signal = interesting_segments(
  f'{SOUND_FILES}/city/ambience/street-ambience.wav',
  base_block=1,
  base_step=0.05
)

# plot em
fig, axs = plt.subplots(2)
axs[0].set_title('original signal')
axs[0].plot(signal)
axs[1].set_title('denoised signal')
axs[1].plot(denoise(signal))
axs[2].set_title('interesting blocks')
axs[2].plot([block[2] for block in blocks])
plt.show()