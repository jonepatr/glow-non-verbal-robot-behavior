tensorboard --logdir /runs &
python server.py &
python train.py hparams/speech2face.json speech2face /data
