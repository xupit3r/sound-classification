init:
	pip install -r requirements.txt

train:
	TF_CPP_MIN_LOG_LEVEL=3 python3 sound.py
