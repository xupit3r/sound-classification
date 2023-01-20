import os
import numpy as np
import soundfile as sf
from sonic.features import get_energies

def segment_audio(sound_file='', seconds_per_segment=1, write_files=False, segments_dir=''):
  signal, sample_rate = sf.read(sound_file)
  signal = signal / (2**15)
  signal_length = len(signal)
  segment_size = seconds_per_segment * sample_rate
  segments = np.array([signal[x:x + segment_size] for x in np.arange(0, signal_length, segment_size)])

  if write_files:
    base_name = os.path.splitext(os.path.basename(sound_file))[0]
    for step, segment in enumerate(segments):
      sf.write(
        f'{segments_dir}/{base_name}_{segment_size * step}_{segment_size * (step + 1)}.wav',
        segment,
        sample_rate
      )
  
  return segments, sample_rate

def remove_silence(segments, sample_rate, outfile=''):
  energies = get_energies(segments)
  threshold = 0.5 * np.median(energies)
  index_of_segments_to_keep = (np.where(energies > threshold)[0])
  segments_to_keep = segments[index_of_segments_to_keep]
  cleaned_signal = np.concatenate(segments_to_keep)

  if len(outfile):
    sf.write(outfile, cleaned_signal, sample_rate)

  return cleaned_signal

def denoise(signal, scale=0.95):
  return np.append(signal[0], signal[1:] - scale * signal[:-1])

def signal_frames(signal, sample_rate, size=0.025, stride=0.01):
  frame_length = int(round(sample_rate * size))
  frame_step = int(round(sample_rate * stride))
  signal_length = len(signal)

  num_frames = int(
    np.ceil(
      float(np.abs(signal_length - frame_length)) 
      / frame_step
    )
  )

  # pad the original signal so that each from is equal length
  pad_signal_length = num_frames * frame_step + frame_length
  z = np.zeros((pad_signal_length - signal_length))
  pad_signal = np.append(signal, z)

  # build the frame indices
  base_index = np.tile(np.arange(0, frame_length), (num_frames, 1))
  frame_index = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
  indices = base_index + frame_index

  # return the frames
  return pad_signal[indices.astype(np.int32, copy=False)]