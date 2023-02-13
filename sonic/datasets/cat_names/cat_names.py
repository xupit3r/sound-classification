"""cat_names dataset."""
import soundfile as sf
import tensorflow_datasets as tfds
from librosa.util import fix_length


class CatNames(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cat_names dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Audio(file_format="wav", sample_rate=16000),
                    "label": tfds.features.ClassLabel(names=["ada", "eugene"]),
                }
            ),
            supervised_keys=("audio", "label"),
            homepage="https://thejoeshow.net/kitties",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract("https://thejoeshow.net/kitties.zip")

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
