#!/bin/bash
trap "kill 0" EXIT

# this is a script that I use for benchmarking my machine  -kris

for i in `seq 9`; do
    CUDA_VISIBLE_DEVICES=$(($i % 3)) python3 machine_benchmark.py &
    sleep 1
done
wait
