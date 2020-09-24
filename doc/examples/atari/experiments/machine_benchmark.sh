#!/bin/bash
trap "kill 0" EXIT

# this is a script that I use for benchmarking my machine  -kris

CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &
CUDA_VISIBLE_DEVICES=2 python3 machine_benchmark.py &

wait
