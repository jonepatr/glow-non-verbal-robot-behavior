import argparse

from glow.builder import build
from glow.config import JsonConfig
from glow.dataset import Speech2FaceDataset
from glow.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="hparams/speech2face_gpu.json")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    hparams = JsonConfig(args.config)

    # build graph and dataset
    built = build(hparams, True)

    train_files = hparams.Files.train
    validation_files = hparams.Files.validation

    if args.small:
        train_files = train_files[:2]
        validation_files = validation_files[:2]

    train_dataset = Speech2FaceDataset(
        train_files,
        data_dir=hparams.Dir.data,
        total_frames=hparams.Glow.image_shape[0],
        audio_feature_type=hparams.Data.audio_feature_type,
    )

    validation_dataset = Speech2FaceDataset(
        validation_files,
        data_dir=hparams.Dir.data,
        total_frames=hparams.Glow.image_shape[0],
        audio_feature_type=hparams.Data.audio_feature_type,
    )

    # begin to train
    trainer = Trainer(
        **built,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        hparams=hparams
    )
    trainer.train()
