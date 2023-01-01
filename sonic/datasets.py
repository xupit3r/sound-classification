from sklearn.model_selection import train_test_split
from operator import itemgetter
import os as os
import csv as csv
import librosa as librosa
import cv2 as cv2
import numpy as np

DATASETS_DIR = '.datasets'
BINARY_OUTPUT = '.cached_datasets'

def read_dataset_info(name='', file_key='', class_key='', classname_key=''):
  with open(f'{DATASETS_DIR}/{name}/config.csv') as file:
    reader = csv.DictReader(file, delimiter=',')

    info = []
    for row in reader:
      info.append([row[file_key], int(row[class_key]),  row[classname_key]])

    class_names = get_classnames(info)
    num_classes = len(class_names)

    return info, num_classes, class_names

def read_wav_files(dataset_name='', dir='', height=173, width=40):
  if len(dir) == 0:
    dir = 'sounds'

  files = os.listdir(f'{DATASETS_DIR}/{dataset_name}/{dir}')

  sound_files = {}
  for file in files:
    raw, sample_rate  = librosa.load(f'{DATASETS_DIR}/{dataset_name}/{dir}/{file}', res_type='kaiser_fast')
    audio = librosa.feature.mfcc(y=raw, sr=sample_rate, n_mfcc=40)
    sound_files[file] = cv2.resize(audio, (width, height), interpolation=cv2.INTER_LINEAR)

  return sound_files

def get_classnames(info):
  return list(set(map(itemgetter(-1), info)))

def to_one_hot(label, dimension=10):
  one_hot = np.zeros((dimension,))
  one_hot[label] = 1.
  return one_hot

def build_urban_sounds(height=173, width=40):
  info, num_classes, class_names = read_dataset_info(
    name='urban_sounds', 
    file_key='slice_file_name',
    class_key='classID',
    classname_key='class'
  )

  print('processing sound files')
  sound_files = read_wav_files(
    dataset_name='urban_sounds',
    dir='sounds',
    height=height,
    width=width
  )

  x = []
  y = []
  for pair in info:
    x.append(sound_files[pair[0]])
    y.append(to_one_hot(pair[1], num_classes))
  
  x = np.array(x)
  x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
  y = np.array(y)

  # save these out to the filesystem for reuse
  np.savez(f'{BINARY_OUTPUT}/urban_sounds.npz', x, y)

def get_urban_sounds():
    info, num_classes, class_names = read_dataset_info(
      name='urban_sounds', 
      file_key='slice_file_name',
      class_key='classID',
      classname_key='class'
    )

    npz = np.load(f'{BINARY_OUTPUT}/urban_sounds.npz')
    x = npz['arr_0']
    y = npz['arr_1']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    return (train_x, train_y), (test_x, test_y), class_names
