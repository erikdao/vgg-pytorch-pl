"""
Main entry to the program: training/inference
"""
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import VGGNet
from data import create_dataloader

pl.seed_everything(1)


def main():
    # create the model
    model = VGGNet(num_classes=10)

    # create data loader
    train_loader = create_dataloader('CIFAR10', './dataset/cifar10', split='train', batch_size=256)
    val_loader = create_dataloader('CIFAR10', './dataset/cifar10', split='val', batch_size=256, shuffle=False)

    # create the trainer
    trainer = Trainer(
        max_epochs=1000, gpus=1, auto_select_gpus=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()