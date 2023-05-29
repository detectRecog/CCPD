# standard library imports
import argparse

# third party imports
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
from data.datasets import PretrainDataset
from modules.detection import DetectionModule

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=300)
parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction)
parser.add_argument("--devices", type=int, nargs="+", default=1)
args = parser.parse_args()

if __name__ == "__main__":
    # defining callbacks
    # 1. checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="pretrain_loss",
        filename="detection-{epoch:02d}-{pretrain_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # 2. learning rate monitor callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # defining the model
    detection_model = DetectionModule(
        dataset=PretrainDataset(split_file=["train.txt"]),
        batch_size=args.batch_size,
    )

    trainer = Trainer(
        # fast_dev_run=True,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_logger],
        # limit_train_batches=0.05,
        # limit_test_batches=0.05,
        # limit_val_batches=0.05,
        log_every_n_steps=1,
        # logger=WandbLogger(
        #     entity="mtp-ai-board-game-engine",
        #     project="cv-project",
        #     group="pretraining-batch-size=32",
        #     log_model="all",
        # ),
        accelerator="gpu" if args.use_gpu else "cpu",
        devices=args.devices,
    )
    trainer.fit(model=detection_model)
