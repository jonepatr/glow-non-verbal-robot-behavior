"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root>
"""
import os

from torchvision import transforms

import vision
from docopt import docopt
from glow.builder import build
from glow.config import JsonConfig
from glow.trainer import Trainer

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    dataset_root = args["<dataset_root>"]
    assert dataset in vision.Datasets, "`{}` is not supported, use `{}`".format(
        dataset, vision.Datasets.keys()
    )
    # assert os.path.exists(
    #    dataset_root
    # ), "Failed to find root dir `{}` of dataset.".format(dataset_root)
    assert os.path.exists(hparams), "Failed to find hparams josn `{}`".format(hparams)
    hparams = JsonConfig(hparams)
    dataset = vision.Datasets[dataset]
    print(dataset)
    # set transform of dataset
    transform = transforms.Compose(
        [
            transforms.CenterCrop(hparams.Data.center_crop),
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),
        ]
    )
    # build graph and dataset
    built = build(hparams, True)
    dataset = dataset(dataset_root, transform=transform)
    # begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams)
    trainer.train()
