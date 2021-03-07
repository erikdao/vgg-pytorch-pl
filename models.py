"""
Model definitions
"""
import os
from datetime import datetime
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage

def write_log(fpath, **kwargs):
    with open(fpath, 'a') as f:
        f.write(f"{kwargs.get('epoch')},{kwargs.get('lr')}," \
                f"{kwargs.get('train_loss')},{kwargs.get('train_acc')},"
                f"{kwargs.get('val_loss')},{kwargs.get('val_acc')}\n")


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
            ('fc1', nn.Linear(in_features=512 * 8 * 8, out_features=4096)),
            ('relu6', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout()),
            ('fc2', nn.Linear(in_features=4096, out_features=4096)),
            ('relu7', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout()),
            ('fc3', nn.Linear(in_features=4096, out_features=kwargs['num_classes'])),
        ]))

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        self._init_weights()
        self._init_manual_logs()

        # Dictionary to hold training_loss by epochs
        self.train_epoch_loss = dict()
        self.train_epoch_acc = dict()
        self.val_epoch_loss = dict()
        self.val_epoch_acc = dict()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def _init_manual_logs(self):
        """
        Setup manual logging to CSV file for further usage
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(cur_dir, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        cur_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.csv_log_path = os.path.join(logs_dir, f'{cur_time}_logs.csv')
        with open(self.csv_log_path, 'w') as f:
            f.write("epoch,lr,train_loss,train_acc,val_loss,val_acc\n")

    def forward(self, x):
        hid = self.features(x)
        hid = hid.view(hid.shape[0], -1)
        return self.classifiers(hid)


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

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
        # Calculate average training loss
        np_loss = [out['loss'].cpu().detach().numpy() for out in outs]
        avg_loss = np.mean(np_loss)
        self.train_epoch_loss[self.trainer.current_epoch] = avg_loss

        # Train accuracy average for this epoch
        train_acc = self.train_accuracy.compute()
        self.train_epoch_acc[self.trainer.current_epoch] = train_acc.cpu().detach().numpy()
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self, val_batch, val_id):
        inputs, labels = val_batch
        pred = self.forward(inputs)
        loss = F.cross_entropy(pred, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.val_accuracy(F.softmax(pred, dim=-1), labels)
        self.log('val_acc_epoch', self.val_accuracy, on_step=True, on_epoch=True)

        return loss

    def validation_epoch_end(self, outs):
        np_loss = [out.cpu().detach().numpy() for out in outs]
        avg_loss = np.mean(np_loss)
        self.val_epoch_loss[self.trainer.current_epoch] = avg_loss

        # Train accuracy average for this epoch
        val_acc = self.train_accuracy.compute()
        self.val_epoch_acc[self.trainer.current_epoch] = val_acc.cpu().detach().numpy()

        # We only want to log on real training loop, not during vanity check
        # CAUTION: This API is constantly changing with Pytorch Lightning update
        if not self.trainer.running_sanity_check:
            write_log(
                self.csv_log_path,
                epoch=self.trainer.current_epoch,
                lr=self._get_current_lr(),
                train_loss=self.train_epoch_loss[self.trainer.current_epoch],
                train_acc=self.train_epoch_acc[self.trainer.current_epoch],
                val_loss=self.val_epoch_loss[self.trainer.current_epoch],
                val_acc=self.val_epoch_acc[self.trainer.current_epoch]
            )

    def _get_current_lr(self):
        """
        Get current learning rate from the optimizer, asssuming that we
        are using only one optimizer and learning rate scheduler
        """
        scheduler = self.trainer.lr_schedulers[0]['scheduler']
        return scheduler.optimizer.param_groups[0]['lr']

