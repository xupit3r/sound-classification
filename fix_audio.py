import os
import soundfile as sf
import numpy as np
from librosa.util import fix_length
from librosa.effects import pitch_shift
from random import randrange

for subdir, dirs, files in os.walk("sonic/custom_datasets/cat_names"):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(subdir, file)
            signal, sample_rate = sf.read(full_path)
            step = randrange(-5, 5)
            shifted = pitch_shift(signal, sr=sample_rate, n_steps=step)
            noisey = shifted + 0.009 * np.random.normal(0, 1, len(shifted))
            sf.write(full_path, fix_length(shifted, size=10624), sample_rate)
