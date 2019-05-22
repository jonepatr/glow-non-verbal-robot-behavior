#!/bin/bash

sudo nvidia-docker run -it \
  -p 6541:6541 -v $(pwd)/results/speech2face:/logdir \
  glow_speech2face tensorboard --logdir /logdir --port 6541
