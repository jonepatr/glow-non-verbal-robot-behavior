trap 'kill $(jobs -p)' EXIT
tensorboard --logdir /runs --port 6007 &
python server.py &
python train.py hparams/speech2face.json speech2face /data
