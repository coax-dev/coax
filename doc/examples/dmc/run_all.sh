#!/bin/bash
trap "kill 0" EXIT

gio trash -f ./data

for f in $(ls ./*.py); do
    JAX_PLATFORM_NAME=cpu python3 $f &
done

wait
