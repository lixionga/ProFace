HOST='127.0.0.0'
PORT='1234'
NUM_GPU=1

python test_corrector.py \
        --manual_seed 2024 \
        --dist_url tcp://${HOST}:${PORT} \
        --world_size ${NUM_GPU} \
        --rank 0 \
        --detector 'xception'