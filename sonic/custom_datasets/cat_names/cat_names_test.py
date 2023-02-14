"""cat_names dataset."""

import tensorflow_datasets as tfds
from . import cat_names


class CatNamesTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for cat_names dataset."""

    # TODO(cat_names):
    DATASET_CLASS = cat_names.CatNames
    SPLITS = {
        "train": 150,  # Number of fake train example
        "test": 50,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()
