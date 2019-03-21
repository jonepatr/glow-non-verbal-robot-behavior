import os

from torchvision import transforms

from docopt import docopt
from glow.builder import build
from glow.config import JsonConfig
from glow.trainer import Trainer
from glow.dataset import Speech2FaceDataset

"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root>
"""


if __name__ == "__main__":
    # args = docopt(__doc__)
    hparams_json = "hparams/speech2face.json"  # args["<hparams>"]
    # dataset = args["<dataset>"]
    # dataset_root = args["<dataset_root>"]
    # assert dataset in vision.Datasets, "`{}` is not supported, use `{}`".format(
    # dataset, vision.Datasets.keys()
    # )
    # assert os.path.exists(
    #    dataset_root
    # ), "Failed to find root dir `{}` of dataset.".format(dataset_root)
    assert os.path.exists(hparams_json), "Failed to find hparams josn `{}`".format(
        hparams_json
    )
    hparams = JsonConfig(hparams_json)
    # dataset = vision.Datasets[dataset]
    # print(dataset)
    # set transform of dataset
    # transform = transforms.Compose(
    #     [
    #         transforms.CenterCrop(hparams.Data.center_crop),
    #         transforms.Resize(hparams.Data.resize),
    #         transforms.ToTensor(),
    #     ]
    # )
    # build graph and dataset
    built = build(hparams, True)
    dataset = Speech2FaceDataset(
        data_dir=hparams.Dir.data,
        total_frames=hparams.Glow.image_shape[0],
        small=hparams.Data.small,
        audio_feature_type=hparams.Data.audio_feature_type,
    )
    # begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams)
    trainer.train()
