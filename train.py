import os

from glow.builder import build
from glow.config import JsonConfig
from glow.dataset import Speech2FaceDataset
from glow.trainer import Trainer

if __name__ == "__main__":
    hparams = JsonConfig("hparams/speech2face.json")

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
