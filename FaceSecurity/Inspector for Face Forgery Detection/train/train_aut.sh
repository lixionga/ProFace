HOST='127.0.0.0'
PORT='12345'
NUM_GPU=4

python train_corrector.py \
        --manual_seed 2024 \
        --dist_url tcp://${HOST}:${PORT} \
        --world_size ${NUM_GPU} \
        --rank 0 \
        --batchsize 128 \
        --dataset 'FF++' \
        --epoch 50