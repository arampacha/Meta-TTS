import torch
import numpy as np

from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl
from collections import defaultdict
from math import ceil
from typing import Literal

from .utils import DistributedProxySampler
from dataset import XvecDataset as Dataset
from utils.tools import pad_2D


class XvecDataModule(pl.LightningDataModule):
    """Datamodule for Xvec.

    Attributes:
        num_classes: Number of target labels.
    """

    def __init__(self,
                 preprocess_config: dict,
                 train_config: dict,
                 *args,
                 target: Literal["speaker", "region", "accent"] = "speaker"):
        """Initialize XvecDataModule.

        Args:
            preprocess_config:
                A dict at least containing paths required by XvecDataset.
            train_config:
                A dict at least containing batch_size.
            target:
                speaker/region/accent.
        """
        super().__init__()
        self.save_hyperparameters()
        self.preprocess_config = preprocess_config
        self.train_config = train_config

        # Discriminate train/transfer
        self.target = target
        self.dataset = Dataset("all", self.preprocess_config)
        self.num_classes = len(getattr(self.dataset, f"{self.target}_map"))

    def setup(self, stage=None):
        # if stage in (None, 'fit'):
        subsets = self._split(self.dataset)
        self.train_datasets, self.val_datasets, self.test_datasets = subsets

    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.train_dataset = ConcatDataset(self.train_datasets)

        def calculate_weights(subsets):
            class_sample_count = torch.tensor([len(sset) for sset in subsets])
            weight = 1. / class_sample_count.double().sqrt()
            samples_weight = torch.cat(
                [torch.full((len(sset),), weight[i], dtype=weight.dtype)
                 for i, sset in enumerate(subsets)]
            )
            return samples_weight

        weights = calculate_weights(self.train_datasets)
        num_samples = len(self.train_dataset)
        assert len(weights) == num_samples
        sampler = WeightedRandomSampler(
            weights, num_samples, generator=torch.Generator().manual_seed(42)
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            sampler=DistributedProxySampler(sampler),
            drop_last=True,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.train_loader

    def val_dataloader(self):
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.val_dataset = ConcatDataset(self.val_datasets)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.val_loader

    def test_dataloader(self):
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.test_dataset = ConcatDataset(self.test_datasets)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.test_loader

    def _split(self, dataset):
        map_ids = defaultdict(list)
        for idx, tgt in enumerate(getattr(dataset, self.target)):
            map_ids[tgt].append(idx)

        train_sets, val_sets, test_sets = [], [], []
        for tgt in sorted(map_ids):
            total_len = len(map_ids[tgt])
            val_len, test_len = ceil(total_len * 0.3), ceil(total_len * 0.1)
            split_len = [total_len - val_len - test_len, val_len, test_len]
            assert split_len[0] > 0
            train_subset, val_subset, test_subset = random_split(
                Subset(dataset, map_ids[tgt]), split_len,
                generator=torch.Generator().manual_seed(42)
            )
            train_sets.append(train_subset)
            val_sets.append(val_subset)
            test_sets.append(test_subset)
        return train_sets, val_sets, test_sets


def get_single_collate(sort=True):
    """Used with BaselineDataModule"""
    def reprocess(data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        accents = [data[idx]["accent"] for idx in idxs]
        regions = [data[idx]["region"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]

        speakers = np.array(speakers)
        accents = np.array(accents)
        regions = np.array(regions)
        mels = pad_2D(mels)

        return (
            ids,
            torch.from_numpy(speakers).long(),
            torch.from_numpy(accents).long(),
            torch.from_numpy(regions).long(),
            torch.from_numpy(mels).float(),
        )

    def collate_fn(data):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        output = reprocess(data, idx_arr)

        return output

    return collate_fn