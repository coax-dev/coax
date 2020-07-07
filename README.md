[![build](https://github.com/microsoft/coax/workflows/build/badge.svg)](https://github.com/microsoft/coax/actions?query=workflow%3Abuild)
[![docs](https://readthedocs.org/projects/coax/badge/?version=latest)](https://coax.readthedocs.io/en/latest/)

# coax

[![coax](doc/_static/img/coax_logo.png)]()

*Plug-n-Play Reinforcement Learning in Python with [OpenAI Gym](https://gym.openai.com/) and [JAX](https://jax.readthedocs.io/)*


![cartpole.gif](doc/_static/img/cartpole.gif)

Create simple, reproducible RL solutions with JAX function approximators.


## Documentation

For the full documentation, go to [coax.readthedocs.io](https://coax.readthedocs.io/).


## Install

Install using pip:
```
$ pip install -U coax
```
or install from a fresh clone
```
$ git clone https://github.com/microsoft/coax.git
$ pip install -e ./coax
```

N.B. **coax** requires the ``jax`` python package, but it doesn't have an
explicit dependence on it. The reason is that your version of ``jax`` and
``jaxlib`` will depend on your CUDA version.

### Option 1: Install default version of JAX
To install the default version of ``jax`` and ``jaxlib`` you can run
```
$ pip install -U jax jaxlib
```

### Option 2: Install specific version compatible with your CUDA installation
To install a specific version that will be compatible with your CUDA
installation, have a look at the versions listed at:

    https://storage.googleapis.com/jax-releases/

To install a specific version, note down (or copy-paste) the correct version
numbers. For instance, at the time of writing, I'm running Python version 3.6
and my CUDA version is 10.2, so I install this version:
```
$ wget https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.41-cp36-none-linux_x86_64.whl
$ pip install jaxlib-0.1.41-cp36-none-linux_x86_64.whl
$ pip install jax
```

**WARNING** Do not blindly copy-paste this. Either choose Option 1 or figure
out what version is right for your system.

## Examples

To get started, check out the [notebooks](notebooks/) for examples. These are
also included in the documentation:

* https://coax.readthedocs.io/notebooks/

Here's one of the examples from the notebooks, in which we solve the
`CartPole-v0` environment with the SARSA algorithm, using a simple
linear function approximator for our Q-function:


```python
import os

import coax
import gym
import haiku as hk
import jax.numpy as jnp


# set some env vars
os.environ['JAX_PLATFORM_NAME'] = 'cpu'   # tell JAX to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


# the cart-pole MDP
env = gym.make('CartPole-v0')
env = coax.wrappers.TrainMonitor(env, 'data/tensorboard/sarsa')
coax.enable_logging()


class MLP(coax.FuncApprox):
    """ multi-layer perceptron with one hidden layer """
    def body(self, S):
        return jnp.tanh(hk.Linear(4)(S))


# value function and its derived policy
func = MLP(env, random_seed=13, learning_rate=0.02)
q = coax.Q(func)
pi = coax.EpsilonGreedy(q, epsilon=0.1)

# experience tracer
cache = coax.NStepCache(env, n=1, gamma=0.9)

# updater
sarsa = coax.td_learning.Sarsa(q)

# used for early stopping
num_successes = 0


# train
for ep in range(1000):
    s = env.reset()
    pi.epsilon = 0.1 if ep < 25 else 0.01

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # extend last reward as asymptotic best-case return
        if t == env.spec.max_episode_steps - 1:
            assert done
            r = 1 / (1 - cache.gamma)

        cache.add(s, a, r, done)
        while cache:
            transition_batch = cache.pop()
            sarsa.update(transition_batch)

        if done:
            if t == env.spec.max_episode_steps - 1:
                num_successes += 1
            else:
                num_successes = 0
            break

        s = s_next

    if num_successes == 10:
        break


# run env one more time to render
coax.utils.generate_gif(env, pi, filepath="data/sarsa.gif", duration=25)
```

The last episode is rendered, which shows something like the gif at the top of
this page.

