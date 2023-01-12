from sonic.features import extract_features
import os

SOUND_FILES = '.test_data'
SEGMENTS_DIR = f'{SOUND_FILES}/segments'

if not os.path.exists(SEGMENTS_DIR):
  os.makedirs(SEGMENTS_DIR)

sound_file = f'{SOUND_FILES}/city/city-traffic-outdoor.wav'

# retrieve a short term feature vector for the sound file
vector = extract_features(
  sound_file=sound_file
)

print(vector.shape)