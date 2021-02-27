"""
Model definitions
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl


class VGGNet(pl.LightningModule):
    """VGGNet"""
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Layer definition
        self.features = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2)),
            ('batch_norm1_1', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2)),
            ('batch_norm1_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)),
            ('batch_norm2_1', nn.BatchNorm2d(128)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2)),
            ('batch_norm2_2', nn.BatchNorm2d(128)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)),
            ('batch_norm3_1', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2)),
            ('batch_norm3_2', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=2)),
            ('batch_norm3_3', nn.BatchNorm2d(256)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv4_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)),
            ('batch_norm4_1', nn.BatchNorm2d(512)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2)),
            ('batch_norm4_2', nn.BatchNorm2d(512)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=2)),
            ('batch_norm4_3', nn.BatchNorm2d(512)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv5_1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2)),
            ('batch_norm5_1', nn.BatchNorm2d(512)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2)),
            ('batch_norm5_2', nn.BatchNorm2d(512)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=2)),
            ('batch_norm5_3', nn.BatchNorm2d(512)),
            ('relu5_3', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.classifiers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=512 * 8 * 8, out_features=2048)),
            ('relu6', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout()),
            ('fc2', nn.Linear(in_features=2048, out_features=2048)),
            ('relu7', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout()),
            ('fc3', nn.Linear(in_features=2048, out_features=kwargs['num_classes'])),
        ]))
    
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0) 
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        hid = self.features(x)
        hid = hid.view(hid.shape[0], -1)
        return self.classifiers(hid)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}
    
    def training_step(self, train_batch, train_id):
        inputs, labels = train_batch
        pred = self.forward(inputs)
        loss = F.cross_entropy(pred, labels) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_accuracy(F.softmax(pred, dim=-1), labels)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=False)
        return loss 

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self, val_batch, val_id):
        inputs, labels = val_batch
        pred = self.forward(inputs)
        loss = F.cross_entropy(pred, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.val_accuracy(F.softmax(pred, dim=-1), labels)
        self.log('val_acc_epoch', self.val_accuracy, on_step=True, on_epoch=True)
        return loss 
