from sonic.preprocess import segment_audio, remove_silence
import matplotlib.pyplot as plt
import os

SOUND_FILES = '.test_data/sound'
SEGMENTS_DIR = f'{SOUND_FILES}/segments'

if not os.path.exists(SEGMENTS_DIR):
  os.makedirs(SEGMENTS_DIR)

# create segments and write them to individual files
# (this will create 1 second segments)
segments, sample_rate = segment_audio(
  f'{SOUND_FILES}/obama.wav', 
  seconds_per_segment=1,
  write_files=True, 
  segments_dir=SEGMENTS_DIR
)

cleaned_signal = remove_silence(
  segments=segments,
  sample_rate=sample_rate,
  outfile=f'{SOUND_FILES}/obama-silenced-removed.wav'
)

plt.plot(cleaned_signal)
plt.show()

