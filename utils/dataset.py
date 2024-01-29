import gc
import json
import random

import seaborn as sns
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset


def sample_from_lists(lists, n, probabilities, plot=False):
    if len(lists) != len(probabilities):
        raise ValueError("Number of lists should be equal to number of probabilities.")
    chosen_lists = random.choices(lists, probabilities, k=n)
    result = [random.choice(lst) for lst in chosen_lists]
    if plot:
        sns.distplot(chosen_lists)
    return result


class GenericDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            # return_tensors="pt",
        )
        item = {k: torch.tensor(v) for k, v in item.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class GenericDatasetBuilder(object):
    def __init__(
        self,
        dset_path,
        tokenizer_class,
        tokenizer_name,
        device="cuda",
        max_length=1024,
    ):
        self.device = device
        self.data = json.load(open(dset_path))
        self.data = shuffle(self.data)
        gc.collect()
        self.max_length = max_length
        self.tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            # local_files_only=True
        )

    def _extract_texts_labels(self, dataset):
        texts = [i["text"] for i in dataset]
        labels = [i["label"] for i in dataset]
        return texts, labels

    def build(self):
        texts, labels = self._extract_texts_labels(self.data)
        dataset = GenericDataset(
            texts, labels, tokenizer=self.tokenizer, max_length=self.max_length
        )
        del texts, labels
        gc.collect()
        return dataset
