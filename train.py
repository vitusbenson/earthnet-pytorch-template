

import yaml

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from model.channel_net import Channel_Net
from task.stf import STFTask
from task.data import EarthNet2021DataModule

__MODELS__ = {
    "channel_net": Channel_Net
}

def train_model(setting_dict: dict):

    pl.seed_everything(setting_dict["Seed"])
    # Data

    data_args = ["--{}={}".format(key,value) for key, value in setting_dict["Data"].items()]
    data_parser = ArgumentParser()
    data_parser = EarthNet2021DataModule.add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = EarthNet2021DataModule(data_params)

    # Model
    model_args = ["--{}={}".format(key,value) for key, value in setting_dict["Model"].items()]
    model_parser = ArgumentParser()
    model_parser = __MODELS__[setting_dict["Architecture"]].add_model_specific_args(model_parser)
    model_params = model_parser.parse_args(model_args)
    model = __MODELS__[setting_dict["Architecture"]](model_params)

    # Task
    task_args = ["--{}={}".format(key,value) for key, value in setting_dict["Task"].items()]
    task_parser = ArgumentParser()
    task_parser = STFTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = STFTask(model = model, hparams = task_params)

    # Logger
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    # Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='EarthNetScore',
    filename='Epoch-{epoch:02d}-ENS-{EarthNetScore:.4f}',
    save_top_k=-1,
    mode='max',
    period = 5
    )

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["precision"] = 16 if dm.hparams.fp16 else 32
    trainer = pl.Trainer(logger = logger, callbacks = [checkpoint_callback], **trainer_dict)

    dm.setup("fit")
    trainer.fit(task, dm)
    print(f"Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}")


    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
    args = parser.parse_args()

    with open(args.setting, 'r') as fp:
        setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

    train_model(setting_dict)
    