#!/bin/bash
trap "kill 0" EXIT

# CUDA_VISIBLE_DEVICES=0 python3 meta_policy.py &
CUDA_VISIBLE_DEVICES=0 python3 ../dqn_boltzmann.py &
CUDA_VISIBLE_DEVICES=0 python3 ../dqn.py &
CUDA_VISIBLE_DEVICES=0 python3 ../ppo.py &
CUDA_VISIBLE_DEVICES=0 python3 ../ddpg.py &

wait
