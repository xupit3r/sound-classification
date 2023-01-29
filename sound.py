from sonic.datasets import get_urban_sounds
from sonic.models import sound_model_1, sound_model_2
from sonic.train import train

options = {
    "show_summary": True,
    "save_model": True,
    "early_stop": True,
    "plot_results": True,
    "epochs": 20,
    "batch_size": 8,
    "height": 173,
    "width": 40,
    "channels": 1,
}

train(options=options, dataset=get_urban_sounds(), model_builder=sound_model_1)
