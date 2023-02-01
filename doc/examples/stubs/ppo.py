import gymnasium
import coax
import optax
import haiku as hk


# pick environment
env = gymnasium.make(...)
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
v = coax.V(func_v, env)
pi = coax.Policy(func_pi, env)


# slow-moving avg of pi
pi_behavior = pi.copy()


# specify how to update policy and value function
ppo_clip = coax.policy_objectives.PPOClip(pi, optimizer=optax.adam(0.001))
simple_td = coax.td_learning.SimpleTD(v, optimizer=optax.adam(0.001))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


for ep in range(100):
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_behavior(s, return_logp=True)
        s_next, r, done, truncated, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done or truncated, logp)
        while tracer:
            buffer.add(tracer.pop())

        # update
        if len(buffer) == buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):  # ~4 passes
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = ppo_clip.update(transition_batch, td_error)
                env.record_metrics(metrics_v)
                env.record_metrics(metrics_pi)

            buffer.clear()
            pi_behavior.soft_update(pi, tau=0.1)

        if done or truncated:
            break

        s = s_next
