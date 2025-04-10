#!/bin/sh

docker run -it --gpus all --shm-size 64G \
    -v /path/to/this/repository:/app/ \
    sbi bash
