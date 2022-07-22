#!/usr/bin/env python3

import os
import json
import shutil
from glob import glob
from copy import deepcopy

PACKAGEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


nb_template = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%pip install git+https://github.com/coax-dev/coax.git@main --quiet",
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.2"
        }
    },
    "colab": {
        "name": None,
        "provenance": []
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

sdl_videodriver_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run this cell to fix rendering errors.\n",
        "import os\n",
        "os.environ['SDL_VIDEODRIVER'] = 'dummy'",
    ]
}

tensorboard_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%load_ext tensorboard\n",
        "%tensorboard --logdir ./data/tensorboard",
    ]
}


for d_in in glob(os.path.join(PACKAGEDIR, 'doc', 'examples', '*')):
    if not os.path.isdir(d_in) or 'search' in d_in:
        continue

    d_out = d_in.replace('examples', '_notebooks')
    if os.path.exists(d_out):
        shutil.rmtree(d_out)
    os.makedirs(d_out)

    for f_in in glob(f'{d_in}/*.py'):
        f_out = f_in.replace('examples', '_notebooks').replace('.py', '.ipynb')
        nb = deepcopy(nb_template)

        with open(f_in) as r, open(f'{f_out}', 'w') as w:
            lines = list(r)
            nb['colab']['name'] = os.path.split(f_out)[1]
            nb['cells'][-1]['source'] = lines
            if any(("CartPole-v" in line or "Pendulum-v" in line) for line in lines):
                nb['cells'].insert(1, sdl_videodriver_cell)
            if any("tensorboard_dir=" in line for line in lines):
                nb['cells'].insert(1, tensorboard_cell)
            if 'atari' in f_in:
                nb['accelerator'] = 'GPU'
            json.dump(nb, w, indent=1)

        print(f"converted: {f_in} --> {f_out}")
