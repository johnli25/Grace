#!/usr/bin/env bash
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python grace_receiver.py \
  --output grace_receiver_frames \
  --port    9000 \