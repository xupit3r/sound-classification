init:
	pip3.10 install -r requirements.txt

train:
	TF_CPP_MIN_LOG_LEVEL=3 python3 sound.py

sandbox:
	TF_CPP_MIN_LOG_LEVEL=3 python3 sandbox.py

build_datasets:
	TF_CPP_MIN_LOG_LEVEL=3 python3 build_datasets.py

example:
	TF_CPP_MIN_LOG_LEVEL=3 python3 example.py

util:
	python3 util.py