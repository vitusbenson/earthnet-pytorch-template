

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

def test_model(setting_dict: dict, checkpoint: str):


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
    model

    # Task
    task_args = ["--{}={}".format(key,value) for key, value in setting_dict["Task"].items()]
    task_parser = ArgumentParser()
    task_parser = STFTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = STFTask(model = model, hparams = task_params)
    task.load_from_checkpoint(checkpoint_path= checkpoint, context_length = setting_dict["Task"]["context_length"], target_length = setting_dict["Task"]["target_length"], model = model, hparams = task_params)

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["precision"] = 16 if dm.hparams.fp16 else 32
    trainer = pl.Trainer(**trainer_dict)

    dm.setup("test")
    trainer.test(model = task, datamodule=dm, ckpt_path = None)



    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
    parser.add_argument('checkpoint', type = str, metavar='path/to/checkpoint', help='checkpoint file')
    parser.add_argument('track', type = str, metavar='iid|ood|ex|sea', help='which track to test: either iid, ood, ex or sea')
    parser.add_argument('--pred_dir', type = str, default = None, metavar = 'path/to/predictions/directory/', help = 'Path where to save predictions')
    args = parser.parse_args()

    with open(args.setting, 'r') as fp:
        setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

    setting_dict["Task"]["context_length"] = 10 if args.track in ["iid", "ood"] else 20 if args.track == "ex" else 70 if args.track == "sea" else 10
    setting_dict["Task"]["target_length"] = 20 if args.track in ["iid", "ood"] else 40 if args.track == "ex" else 140 if args.track == "sea" else 20

    setting_dict["Data"]["test_track"] = args.track

    if args.pred_dir is not None:
        setting_dict["Task"]["pred_dir"] = args.pred_dir

    test_model(setting_dict, args.checkpoint)
    