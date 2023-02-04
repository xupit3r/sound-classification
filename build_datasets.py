from sonic.datasets import build_urban_sounds
from sonic.features import mfcc, centroid, bandwidth, melspectrogram, contrast

print("building urban sounds dataset...")
build_urban_sounds(feature=mfcc)
