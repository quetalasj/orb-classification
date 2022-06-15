import torch.nn as nn
import torch
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._transform = None
        self.activation = None
        self.loss = None

    def train_backbone(self, requires_grad=True):
        self.backbone.requires_grad_(requires_grad)

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
        # print('images ', images.shape)
        # print('labels ', labels.shape)
        predict = self.activation(self.forward(images))
        # print('predict ', predict.shape)
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
