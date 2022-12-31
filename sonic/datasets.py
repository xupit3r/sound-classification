import csv as csv

DATASETS_DIR = '.datasets'

def read_dataset_config(name='', file_key='', class_key=''):
  with open(f'{DATASETS_DIR}/{name}/config.csv') as file:
    reader = csv.DictReader(file, delimiter=',')

    config = []
    for row in reader:
      config.append([row[file_key], row[class_key]])

    return config

def get_urban_sounds():
  config = read_dataset_config(
    name='urban_sounds', 
    file_key='slice_file_name',
    class_key='classID'
  )

  print(config)

  return ()