import torch.nn as nn
import torch
import pytorch_lightning as pl

class MyLoss(nn.MSELoss):
    def forward(self, input, target):
        return super().forward(input.float(), target.float().unsqueeze(1))
    
class BaseModel(pl.LightningModule):
    def __init__(self, transform, nn_type='classificator'):
        super().__init__()
        self._transform = transform
        self.nn_type = nn_type
        if self.nn_type == 'classificator':
            self.activation = nn.LogSoftmax(dim=1)
            self.loss = nn.NLLLoss()
        elif self.nn_type == 'regressor':
            self.activation = nn.Sigmoid()
            # self.loss = nn.MSELoss()
            self.loss = MyLoss()

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)

    def train_backbone(self):
        self.backbone.requires_grad_(True)

    def prepare_backbone(self, freeze_backbone):
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.train_backbone()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, labels = batch
        predict = self.activation(self.forward(images))
        loss = self.loss(predict, labels)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predict = self.activation(self.forward(images))
        val_loss = self.loss(predict, labels)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
