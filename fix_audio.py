import os
import soundfile as sf
from librosa.util import fix_length

for subdir, dirs, files in os.walk("sonic/datasets/cat_names"):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(subdir, file)
            signal, sample_rate = sf.read(full_path)
            sf.write(full_path, fix_length(signal, size=10624), sample_rate)
