from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import torch
import numpy as np
import os
from pathlib import Path


class ORBClsDataset(Dataset):
    def __init__(self, root, datasets, classification_threshold, transform, combine_data=True):
        root = Path(root)
        images_list = []
        labels_list = []
        features_list = []
        for dataset in datasets:
            print('datasetname:', dataset)
            dataset_path = root / dataset
            images_path = dataset_path / 'images'
            dataset_images = np.array([str(images_path / image) for image in np.sort(os.listdir(images_path))])
            assert len(dataset_images) > 0
            images_list.append(dataset_images)

            dataset_features = pd.read_csv(str(dataset_path / "labels.csv"), header=None).to_numpy().reshape(-1)
            assert dataset_features.shape[0] > 0
            labels_list.append(dataset_features > classification_threshold)
            features_list.append(dataset_features)
            print('num images:', dataset_images.shape[0])
            print('num labels:',  dataset_features.shape[0])
            assert dataset_images.shape[0] == dataset_features.shape[0]

        if combine_data:
            self.images_list = np.concatenate(images_list, axis=0)
            self.labels = np.concatenate(labels_list, axis=0)
        else:
            self.images_list = images_list
            self.labels = labels_list
        self.features_list = features_list

        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.images_list[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
