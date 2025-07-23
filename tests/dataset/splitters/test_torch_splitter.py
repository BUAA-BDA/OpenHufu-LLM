import json
import unittest
from torch.utils.data import Dataset
from openhufu.dataset.splitters.custom.torch_splitter import TorchIIDSplitter, TorchLDASplitter
from openhufu.utils import download_url, get_file_path_without_name
from datasets import load_dataset

class TestTorchSplitter(unittest.TestCase):

    def setUp(self):
        file_path = download_url(
            'https://raw.githubusercontent.com/sahil280114/codealpaca/refs/heads/master/data/rosetta_alpaca.json',
            '../../../dataset/')
        self.dataset = load_dataset(get_file_path_without_name(file_path))['train'].with_format('torch')
        self.n_clients = 3

    def test_iid_split(self):
        splitter = TorchIIDSplitter(n_clients=self.n_clients, seed=42)
        splits = splitter(self.dataset)
        self.assertEqual(len(splits), self.n_clients)

    def test_lda_split(self):
        splitter = TorchLDASplitter(n_clients=self.n_clients, seed=42)
        splits = splitter(self.dataset, get_labels_fn=lambda x: x['input'], alpha=0.8)
        for i, split in enumerate(splits):
            split_data = [split.dataset[item] for item in split.indices]
            with open(f'../../../dataset/rosetta_alpaca/rosetta_alpaca_split_{i}.json', 'w') as file:
                json.dump(split_data, file)
        self.assertEqual(len(splits), self.n_clients)


if __name__ == '__main__':
    unittest.main()
