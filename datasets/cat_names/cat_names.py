"""cat_names dataset."""

import tensorflow_datasets as tfds


class CatNames(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cat_names dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cat_names): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "audio": tfds.features.Audio(shape=(None, None, 1)),
                    "label": tfds.features.ClassLabel(names=["ada", "eugene"]),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("audio", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(cat_names): Downloads the data and defines the splits
        path = dl_manager.download_and_extract("https://todo-data-url")

        # TODO(cat_names): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "train_data"),
            "test": self._generate_examples(path / "test_data"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for audio in path.glob("*.wav"):
            yield audio.name, {
                "audio": audio,
                "label": "eugene" if audio.name.startswith("eugene_") else "ada",
            }
