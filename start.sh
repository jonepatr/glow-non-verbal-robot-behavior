trap 'kill $(jobs -p)' EXIT
tensorboard --logdir /runs --port 6006 &
python server.py &
python train.py hparams/speech2face_gpu.json
