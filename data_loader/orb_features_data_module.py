from torch.utils.data import DataLoader, random_split
from .orb_features_dataset import ORBFeaturesDataset
import pytorch_lightning as pl


class ORBFeaturesDataModule(pl.LightningDataModule):
    def __init__(self,
                 root,
                 dataset,
                 classification_threshold,
                 transform,
                 batch_size=128,
                 shuffle=True,
                 num_workers=4,
                 num_val=1000,
                 num_test=1000,
                 nn_type='classificator',
                 sig_coeff=1):
        super().__init__()
        self.root = root
        self.dataset = dataset
        self.transform = transform

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.classification_threshold = classification_threshold
        self.nn_type = nn_type
        self.sig_coeff = sig_coeff

        self.num_val = num_val
        self.num_test = num_test

    def setup(self, stage):
        dataset = ORBFeaturesDataset(self.root,
                                     self.dataset,
                                     classification_threshold=self.classification_threshold,
                                     transform=self.transform, nn_type = self.nn_type, sig_coeff=self.sig_coeff)
        num_train = len(dataset) - self.num_val - self.num_test
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [num_train,
                                                                                         self.num_val,
                                                                                         self.num_test])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
    #     # Used to clean-up when the run is finished
    #     ...