import gymnasium
import coax
import optax
import haiku as hk


# pick environment
env = gymnasium.make(...)
env = coax.wrappers.TrainMonitor(env)


def torso(S, is_training):
    # custom haiku function for the shared preprocessor
    with hk.experimental.name_scope('torso'):
        net = hk.Sequential([...])
    return net(S)


def func_v(S, is_training):
    # custom haiku function
    X = torso(S, is_training)
    with hk.experimental.name_scope('v'):
        value = hk.Sequential([...])
    return value(X)  # output shape: (batch_size,)


def func_pi(S, is_training):
    # custom haiku function (for discrete actions in this example)
    X = torso(S, is_training)
    with hk.experimental.name_scope('pi'):
        logits = hk.Sequential([...])
    return {'logits': logits(X)}  # logits shape: (batch_size, num_actions)


# function approximators
v = coax.V(func_v, env)
pi = coax.Policy(func_pi, env)


# specify how to update policy and value function
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(0.001))
simple_td = coax.td_learning.SimpleTD(v, optimizer=optax.adam(0.002))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


for ep in range(100):
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        s_next, r, done, truncated, info = env.step(a)

        # add transition to buffer
        # N.B. vanilla-pg doesn't use logp but we include it to make it easy to
        # swap in another policy updater that does require it, e.g. ppo-clip
        tracer.add(s, a, r, done or truncated, logp)
        while tracer:
            buffer.add(tracer.pop())

        # update
        if len(buffer) == buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):  # ~4 passes
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = vanilla_pg.update(transition_batch, td_error)
                env.record_metrics(metrics_v)
                env.record_metrics(metrics_pi)

                # optional: sync shared parameters (this is not always optimal)
                pi.params, v.params = coax.utils.sync_shared_params(pi.params, v.params)

            buffer.clear()

        if done or truncated:
            break

        s = s_next
