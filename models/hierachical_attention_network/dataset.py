# coding=utf-8
"""
@author: Yantong Lai
@date: 07/20/2020
@description: Transform <filename.csv> data file to <torch.Dataset>
"""

import torch
from torch.utils.data import Dataset, DataLoader
from models.hierachical_attention_network.utils import load_ndarray

import torch.nn as nn
import numpy as np


class AgNewsDataset(Dataset):
    def __init__(self, docs_file, labels_file):
        # Load ndarray
        self.docs_array = load_ndarray(docs_file)
        self.labels_array = load_ndarray(labels_file)
        # Transform ndarray to tensor
        self.docs_tensor = torch.from_numpy(self.docs_array).long()
        self.labels_tensor = torch.from_numpy(self.labels_array).long()
        self.labels_tensor = torch.from_numpy(self.labels_array).long()

    def __getitem__(self, idx):
        features = self.docs_tensor[idx]
        label = self.labels_tensor[idx]
        return label, features

    def __len__(self):
        return self.docs_array.shape[0]

