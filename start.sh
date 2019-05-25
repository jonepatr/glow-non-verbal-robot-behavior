trap 'kill $(jobs -p)' EXIT
tensorboard --logdir /runs --port 6541 > /dev/null 2>&1 &
python server.py 5005 > /dev/null 2>&1 &
python train.py hparams/speech2face_gpu.json $@
