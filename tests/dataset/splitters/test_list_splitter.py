import unittest
from openhufu.dataset.splitters.custom.list_splitter import ListIIDSplitter


class TestListSplitter(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            {"text": "sample1", "label": 0},
            {"text": "sample2", "label": 1},
            {"text": "sample3", "label": 0},
            {"text": "sample4", "label": 1},
            {"text": "sample5", "label": 0},
            {"text": "sample6", "label": 1},
        ]
        self.n_clients = 3
        self.splitter = ListIIDSplitter(n_clients=self.n_clients, seed=42)

    def test_split(self):
        splits = self.splitter(self.dataset)
        self.assertEqual(len(splits), self.n_clients)


if __name__ == "__main__":
    unittest.main()
