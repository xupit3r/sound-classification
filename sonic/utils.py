import os


def get_sound_dirs(root=""):
    return list(map(lambda d: f"{root}/{d}", os.listdir(root)))
