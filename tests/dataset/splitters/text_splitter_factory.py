import unittest
from datasets import Dataset
from openhufu.dataset.splitters.splitter_factory import SplitterFactory
from openhufu.dataset.splitters.custom.hf_splitter import HFIIDSplitter, HFLDASplitter


class TestSplitterFactory(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_dict(
            {"text": ["sample1", "sample2"], "label": [0, 1]}
        )

    def test_get_iid_splitter(self):
        splitter = SplitterFactory.get_splitter(
            type(self.dataset), "iid", n_clients=3, seed=42
        )
        self.assertIsInstance(splitter, HFIIDSplitter)

    def test_get_lda_splitter(self):
        splitter = SplitterFactory.get_splitter(
            type(self.dataset), "lda", n_clients=3, seed=42
        )
        self.assertIsInstance(splitter, HFLDASplitter)

    def test_invalid_splitter_type(self):
        with self.assertRaises(ValueError):
            SplitterFactory.get_splitter(
                type(self.dataset), "invalid", n_clients=3, seed=42
            )

    def test_invalid_dataset_type(self):
        with self.assertRaises(ValueError):
            SplitterFactory.get_splitter(None, "iid", n_clients=3, seed=42)


if __name__ == "__main__":
    unittest.main()
