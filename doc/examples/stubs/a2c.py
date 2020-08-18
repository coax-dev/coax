import gym
import coax
import optax
import haiku as hk


# pick environment
env = gym.make(...)
env = coax.wrappers.TrainMonitor(env)


def func_v(S, is_training):
    # custom haiku function
    value = hk.Sequential([...])
    return value(S)  # output shape: (batch_size,)


def func_pi(S, is_training):
    # custom haiku function (for discrete actions in this example)
    logits = hk.Sequential([...])
    return {'logits': logits(S)}  # logits shape: (batch_size, num_actions)


# function approximators
v = coax.V(func_v, env.observation_space)
pi = coax.Policy(func_pi, env.observation_space, env.action_space)


# specify how to update policy and value function
vanillapg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(0.001))
simple_td = coax.td_learning.SimpleTD(v, optimizer=optax.adam(0.002))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


for ep in range(100):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # add transition to buffer
        # N.B. vanilla-pg doesn't use logp but we include it to make it easy to
        # swap in another policy updater that does require it, e.g. ppo-clip
        tracer.add(s, a, r, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # update
        if len(buffer) == buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):  # ~4 passes
                transition_batch = buffer.sample(batch_size=32)
                td_error = simple_td.td_error(transition_batch)
                vanillapg.update(transition_batch, Adv=td_error)
                simple_td.update(transition_batch)

            buffer.clear()

        if done:
            break

        s = s_next
