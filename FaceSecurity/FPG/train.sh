HOST='127.0.0.0'
PORT='12345'
NUM_GPU=4


python src/train.py \
        --dist_url tcp://${HOST}:${PORT} \
        --world_size ${NUM_GPU} \
        --rank 0 \
        -n fpg