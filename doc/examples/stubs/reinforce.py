import gymnasium
import coax
import optax
import haiku as hk


# pick environment
env = gymnasium.make(...)
env = coax.wrappers.TrainMonitor(env)


def func_pi(S, is_training):
    # custom haiku function (for discrete actions in this example)
    logits = hk.Sequential([...])
    return {'logits': logits(S)}  # logits shape: (batch_size, num_actions)


# function approximator
pi = coax.Policy(func_pi, env)


# specify how to update policy and value function
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(0.001))


# specify how to trace the transitions
tracer = coax.reward_tracing.MonteCarlo(gamma=0.9)


for ep in range(100):
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards to create training data
        # N.B. vanilla-pg doesn't use logp but we include it to make it easy to
        # swap in another policy updater that does require it, e.g. ppo-clip
        tracer.add(s, a, r, done or truncated, logp)

        # update
        while tracer:
            transition_batch = tracer.pop()
            Gn = transition_batch.Rn  # 'Rn' is full return 'Gn' in MC-cache
            metrics = vanilla_pg.update(transition_batch, Adv=Gn)
            env.record_metrics(metrics)

        if done or truncated:
            break

        s = s_next
