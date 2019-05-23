#!/bin/bash

sudo nvidia-docker run -it \
  -p 5005:5005 -v $(pwd):/workspace -v $(pwd)/results/speech2face:/runs \
  glow_speech2face python server.py 5005
