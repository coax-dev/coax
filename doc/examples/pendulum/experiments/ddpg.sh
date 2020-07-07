#!/bin/bash
trap "kill 0" EXIT

# python3 ddpg \
#     --max_num_episodes 1000 \
#     --learning_rate 0.02 \
#     --logvar 0 \
#     --log_transform \
#     --kl_div_beta 0.01 \
#     --gamma 0.9 \
#     --n_step 1 \
#     --sync_tau 0.5 \
#     --sync_period 20

for learning_rate in 0.1 0.01 0.001; do
for kl_div_beta in 0.1 0.01 0.001; do
python3 ddpg.py \
    --max_num_episodes 1000 \
    --learning_rate $learning_rate \
    --logvar 0 \
    --log_transform \
    --kl_div_beta $kl_div_beta \
    --gamma 0.9 \
    --n_step 5 \
    --sync_tau 0.5 \
    --sync_period 20 &
done
done | tee data/ddpg.log
wait
