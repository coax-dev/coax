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
    def body(self, S, is_training):
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
