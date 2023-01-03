init:
	pip3.10 install -r requirements.txt

train:
	TF_CPP_MIN_LOG_LEVEL=3 python3 sound.py

build_datasets:
	TF_CPP_MIN_LOG_LEVEL=3 python3 build_datasets.py