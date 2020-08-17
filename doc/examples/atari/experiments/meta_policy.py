import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix
from ray.rllib.env.atari_wrappers import wrap_deepmind


# set some environment variable
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
gifs_filepath = "data/gifs/meta/T{:08d}.gif"
coax.enable_logging('meta')


# env with preprocessing
ENV = gym.make('PongNoFrameskip-v4')  # wrap_deepmind will do frame skipping
ENV = wrap_deepmind(ENV)


class Func(coax.FuncApprox):
    def body(self, S, is_training):
        M = coax.utils.diff_transform_matrix(num_frames=S.shape[-1])
        seq = hk.Sequential([  # S.shape = [batch, h, w, num_stack]
            lambda x: jnp.dot(S / 255, M),  # [b, h, w, n]
            hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
            hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256), jax.nn.relu,
        ])
        return seq(S)

    def optimizer(self):
        return optix.adam(learning_rate=0.00025)


class DQN1:
    env = coax.wrappers.TrainMonitor(
        ENV, tensorboard_dir="data/tensorboard/meta_dqn_epsilongreedy",
        tensorboard_write_all=True)

    # function approximators
    func = Func(env)
    q = coax.Q(func, qtype=2)
    q_targ = q.copy()
    pi = coax.EpsilonGreedy(q, epsilon=1.)

    # updater
    qlearning = coax.td_learning.QLearning(q, q_targ)

    # replay buffer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)

    # DQN exploration schedule (stepwise linear annealing)
    @staticmethod
    def epsilon(T):
        M = 1000000
        if T < M:
            return 1 - 0.9 * T / M
        if T < 2 * M:
            return 0.1 - 0.09 * (T - M) / M
        return 0.01


class DQN2:
    env = coax.wrappers.TrainMonitor(
        ENV, tensorboard_dir="data/tensorboard/meta_dqn_boltzmann",
        tensorboard_write_all=True)

    # function approximators
    func = Func(env)
    q = coax.Q(func, qtype=2)
    q_targ = q.copy()
    pi = coax.BoltzmannPolicy(q, tau=0.015)

    # updater
    qlearning = coax.td_learning.QLearning(q, q_targ)

    # replay buffer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


class DDPG:
    env = coax.wrappers.TrainMonitor(
        ENV, tensorboard_dir="data/tensorboard/meta_ddpg",
        tensorboard_write_all=True)

    # use separate function approximators for pi and q
    pi = coax.Policy(Func(env))
    q = coax.Q(Func(env))

    # target networks
    pi_targ = pi.copy()
    q_targ = q.copy()

    # policy regularizer (avoid premature exploitation)
    kl = coax.policy_regularizers.KLDivRegularizer(pi, beta=0.001)

    # updaters
    determ_pg = coax.policy_objectives.DeterministicPG(pi, q, regularizer=kl)
    qlearning = coax.td_learning.QLearningMode(q, pi_targ, q_targ)

    # replay buffer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


class PPO:
    env = coax.wrappers.TrainMonitor(
        ENV, tensorboard_dir="data/tensorboard/meta_ppo",
        tensorboard_write_all=True)

    # function approximators (using two separate function approximators)
    func_pi, func_v = Func(env), Func(env)
    pi = coax.Policy(func_pi)
    pi_old = pi.copy()
    v = coax.V(func_v)
    v_targ = v.copy()

    # we'll use this to temporarily store our experience
    tracer = coax.reward_tracing.NStep(n=5, gamma=0.99)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)

    # policy regularizer (avoid premature exploitation)
    kl = coax.policy_regularizers.KLDivRegularizer(pi, beta=0.001)

    # updaters
    simple_td = coax.td_learning.SimpleTD(v, v_targ)
    ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=kl)


class META:  # based on DQN1
    # polcies or "arms" of the contextual bandit
    arms = (
        # exploring arms
        DQN1.pi,  # eps-greedy
        DQN2.pi,  # boltzmann
        DDPG.pi,
        PPO.pi_old,
        # exploiting arms
        DQN1.pi.greedy,
        DQN2.pi.greedy,
        DDPG.pi.greedy,
        PPO.pi_old.greedy,
    )
    env = coax.wrappers.MetaPolicyEnv(ENV, *arms)
    env = coax.wrappers.TrainMonitor(
        env, tensorboard_dir="data/tensorboard/meta_bandit",
        tensorboard_write_all=True)

    # function approximators
    func = Func(env)
    q = coax.Q(func, qtype=2)
    q_targ = q.copy()
    pi = coax.BoltzmannPolicy(q, tau=0.015)

    # updater
    qlearning = coax.td_learning.QLearning(q, q_targ)

    # replay buffer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=10000)


while META.env.T < 3000000:
    DQN1.pi.epsilon = DQN1.epsilon(META.env.T)  # epsilon-greedy schedule

    s = META.env.reset()

    for t in range(5000):
        a_meta, logp_meta = META.pi(s, return_logp=True)
        s_next, r, done, info = META.env.step(a_meta)

        # extract low-level actions and their log-propensities
        a, logp = info['a'], info['logp']

        # sync counters
        DQN1.env.T = DQN2.env.T = DDPG.env.T = PPO.env.T = META.env.T
        DQN1.env.ep = DQN2.env.ep = DDPG.env.ep = PPO.env.ep = META.env.ep

        # -- DQN1 updates -----------------------------------------------------
        DQN1.tracer.add(s, a, r, done, logp)
        while DQN1.tracer:
            DQN1.buffer.add(DQN1.tracer.pop())

        if len(DQN1.buffer) > 50000:  # buffer warm-up
            DQN1.qlearning.update(DQN1.buffer.sample(batch_size=32))

        if META.env.T % 10000 == 0:
            DQN1.q_targ.soft_update(DQN1.q, tau=1)

        # -- DQN2 updates -----------------------------------------------------
        DQN2.tracer.add(s, a, r, done, logp)
        while DQN2.tracer:
            DQN2.buffer.add(DQN2.tracer.pop())

        if len(DQN2.buffer) > 50000:  # buffer warm-up
            DQN2.qlearning.update(DQN2.buffer.sample(batch_size=32))

        if META.env.T % 10000 == 0:
            DQN2.q_targ.soft_update(DQN2.q, tau=1)

        # -- DDPG updates -----------------------------------------------------
        DDPG.tracer.add(s, a, r, done, logp)
        while DDPG.tracer:
            DDPG.buffer.add(DDPG.tracer.pop())

        if len(DDPG.buffer) > 50000:  # buffer warm-up
            transition_batch = DDPG.buffer.sample(batch_size=32)
            DDPG.determ_pg.update(transition_batch)
            DDPG.qlearning.update(transition_batch)

        if META.env.T % 10000 == 0:
            DDPG.pi_targ.soft_update(DDPG.pi, tau=1)
            DDPG.q_targ.soft_update(DDPG.q, tau=1)

        # -- PPO updates ------------------------------------------------------
        PPO.tracer.add(s, a, r, done, logp)
        while PPO.tracer:
            PPO.buffer.add(PPO.tracer.pop())

        if len(PPO.buffer) >= PPO.buffer.capacity:
            num_batches = int(4 * PPO.buffer.capacity / 32)
            for _ in range(num_batches):
                transition_batch = PPO.buffer.sample(32)
                Adv = PPO.simple_td.td_error(transition_batch)
                PPO.ppo_clip.update(transition_batch, Adv)
                PPO.simple_td.update(transition_batch)
            PPO.buffer.clear()

            # sync target networks
            PPO.pi_old.soft_update(PPO.pi, tau=0.1)
            PPO.v_targ.soft_update(PPO.v, tau=0.1)

        # -- META-policy updates ----------------------------------------------
        META.tracer.add(s, a, r, done, logp)
        while META.tracer:
            META.buffer.add(META.tracer.pop())

        if len(META.buffer) > 50000:  # buffer warm-up
            META.qlearning.update(META.buffer.sample(batch_size=32))

        if META.env.T % 10000 == 0:
            META.q_targ.soft_update(META.q, tau=1)

        # ---------------------------------------------------------------------

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if META.env.period(name='generate_gif', T_period=10000):
        T = META.env.T - META.env.T % 10000
        coax.utils.generate_gif(
            env=META.env, policy=META.pi.greedy, resize_to=(320, 420),
            filepath=gifs_filepath.format(T))
