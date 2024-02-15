# standard library imports
import os
import argparse

# 3rd party imports
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
from modules.recognition import RecognitionModule
from data.datasets import TrainDataset, TestDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--pretrained_model_path", type=str, default=None)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction)
parser.add_argument("--devices", type=int, nargs="+", default=1)
args = parser.parse_args()


if __name__ == "__main__":
    model = RecognitionModule(
        train_set=TrainDataset(),
        val_set=TestDataset(["val.txt"]),
        test_set=TestDataset(["test.txt"]),
        batch_size=args.batch_size,  # 16 is the default, modify according to available GPU memory
        pretrained_model_path=args.pretrained_model_path,  # filename of the pretrained model (in src/logs),
        # make sure this file exists in the logs folder on the machine you're running on
    )

    # initialise ModelCheckpoint Callback, which saves the top 3 models based
    # on validation precision (no need to modify)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_precision",
        filename="recognition-{epoch:02d}-{val_precision:.2f}",
        save_top_k=3,
        mode="min",
    )

    # init learning rate callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    trainer = Trainer(
        max_epochs=args.max_epochs,  # number of epochs to train for
        # precision=16,
        callbacks=[
            checkpoint_callback,
            lr_logger,
        ],  # add the checkpoint callback, no need to modify
        default_root_dir=os.getenv(
            "LOG_DIR"
        ),  # path to the logs folder, no need to modify
        strategy="ddp_find_unused_parameters_false",  # no need to modify
        log_every_n_steps=1,  # logging interval, no need to modify
        accelerator="gpu"
        if args.use_gpu
        else "cpu",  # modify this based on the machine you're running on
        devices=args.devices,  # device indices for the GPUs
    )

    trainer.fit(model=model)

    trainer.test(ckpt_path="best")
