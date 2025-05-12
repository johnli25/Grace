#!/usr/bin/env bash
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python grace_receiver.py \
  --ip 0.0.0.0 \
  --port    9000 \
  --deadline_ms 1300 