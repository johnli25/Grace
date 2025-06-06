#!/usr/bin/env bash
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python grace_sender.py \
  --input ../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Diving-Side001.mp4 \
  --ip      127.0.0.1 \
  --port    9000 \