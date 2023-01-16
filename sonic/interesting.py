import numpy as np
import soundfile as sf
import os

SOUND_FILES = '.test_data'
SEGMENTS_DIR = f'{SOUND_FILES}/segments'

def normalized_energy(block, signal_energy):
  block_energy = (block**2).sum() / len(block)
  return max((block_energy - signal_energy), 0) / signal_energy

def interesting_segments(sound_file='', base_block=1, base_step=0.05):
  signal, sample_rate = sf.read(sound_file)
  signal_length = len(signal)
  signal_energy = (signal**2).sum() / signal_length

  block_size = int(base_block * sample_rate)
  step_size = int(base_step * sample_rate)

  blocks = np.array([
    (x, x + block_size, normalized_energy(signal[x:x + block_size], signal_energy)) 
    for x in np.arange(0, signal_length, step_size)
  ], dtype=object)

  segments = []
  start = 0
  for block in blocks:
    if (block[2] > 0 and start == 0):
      start = block[0]
    elif (block[2] == 0 and start > 0):
      segments.append(np.array(signal[start:block[0]]))
      start = 0

  base_name = os.path.splitext(os.path.basename(sound_file))[0]

  for step, segment in enumerate(segments):
    sf.write(
      f'{SEGMENTS_DIR}/{base_name}_{step + 1}.wav',
      segment,
      sample_rate
    )

  return segments
