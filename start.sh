trap 'kill $(jobs -p)' EXIT
tensorboard --logdir /runs --port 6000 > /dev/null 2>&1 &
python server.py 5000 > /dev/null 2>&1 &
python train.py hparams/speech2face_gpu.json $@
