import os
import numpy as np
import soundfile as sf

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
  energies = [(s**2).sum() / len(s) for s in segments]
  threshold = 0.5 * np.median(energies)
  index_of_segments_to_keep = (np.where(energies > threshold)[0])
  segments_to_keep = segments[index_of_segments_to_keep]
  cleaned_signal = np.concatenate(segments_to_keep)

  if len(outfile):
    sf.write(outfile, cleaned_signal, sample_rate)

  return cleaned_signal