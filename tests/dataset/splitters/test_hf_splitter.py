import unittest
from datasets import Dataset, load_dataset
from openhufu.dataset.splitters.custom.hf_splitter import HFIIDSplitter, HFLDASplitter
from openhufu.utils import download_url, get_file_path_without_name


class TestHFIIDSplitter(unittest.TestCase):

    def setUp(self):
        data = {
            "text": ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"],
            "label": [0, 1, 0, 1, 0, 1],
        }
        self.dataset = Dataset.from_dict(data)
        self.n_clients = 3
        self.splitter = HFIIDSplitter(n_clients=self.n_clients, seed=42)

    def test_split(self):
        splits = self.splitter(self.dataset)
        self.assertEqual(len(splits), self.n_clients)


class TestHFLDASplitter(unittest.TestCase):

    def setUp(self):
        file_path = download_url(
            "https://raw.githubusercontent.com/sahil280114/codealpaca/refs/heads/master/data/rosetta_alpaca.json",
            "../../../dataset/",
        )
        self.dataset = load_dataset(get_file_path_without_name(file_path))["train"]
        self.n_clients = 3
        self.splitter = HFLDASplitter(n_clients=self.n_clients, seed=42)

    def test_split(self):
        splits = self.splitter(
            self.dataset, get_labels_fn=lambda x: x["input"], alpha=0.8
        )
        import numpy as np

        s = np.unique(splits[0]["input"])
        self.assertEqual(len(splits), self.n_clients)


if __name__ == "__main__":
    unittest.main()
