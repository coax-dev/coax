import os

import coax
import gym
import haiku as hk
import jax


# set some env vars
os.environ['JAX_PLATFORM_NAME'] = 'cpu'   # tell JAX to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


# the cart-pole MDP
env = gym.make('CartPole-v0')
env = coax.wrappers.TrainMonitor(env, 'data/tensorboard/a2c')
coax.enable_logging()


class MLP(coax.FuncApprox):
    """ multi-layer perceptron with one hidden layer """
    def body(self, S, is_training):
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
        ))
        return seq(S)

    def optimizer(self):
        from jax.experimental import optix
        return optix.chain(
            optix.apply_every(k=32),  # update in batches of size k
            optix.adam(**self.optimizer_kwargs))


# value function and its derived policy
func_v = MLP(env, random_seed=13, learning_rate=0.005)
func_pi = MLP(env, random_seed=13, learning_rate=0.001)
v = coax.V(func_v)
pi = coax.Policy(func_pi)

# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)

# updaters
vanilla_pg = coax.policy_objectives.VanillaPG(pi)
value_td = coax.td_learning.ValueTD(v)


# train
for ep in range(1000):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)
        if done and (t == env.spec.max_episode_steps - 1):
            r = 1 / (1 - tracer.gamma)

        tracer.add(s, a, r, done)
        while tracer:
            transition_batch = tracer.pop()
            Adv = value_td.td_error(transition_batch)

            metrics = {}
            metrics.update(vanilla_pg.update(transition_batch, Adv))
            metrics.update(value_td.update(transition_batch))
            env.record_metrics(metrics)

        if done:
            break

        s = s_next

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        break


# run env one more time to render
coax.utils.generate_gif(env, pi, filepath="data/a2c.gif", duration=25)
