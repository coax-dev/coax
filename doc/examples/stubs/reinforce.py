import gym
import coax


# pick environment
env = gym.make(...)
env = coax.wrappers.TrainMonitor(env)


# show logs from TrainMonitor
coax.enable_logging()


class MyFuncApprox(coax.FuncApprox):
    def body(self, S, is_training):
        # custom haiku function
        ...


# define function approximator
func = MyFuncApprox(env)
pi = coax.Policy(func)


# specify how to update policy and value function
vanilla_pg = coax.policy_objectives.VanillaPG(pi)


# specify how to trace the transitions
cache = coax.reward_tracing.MonteCarloCache(env, gamma=0.9)


for ep in range(100):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # add transition to cache
        # N.B. vanilla-pg doesn't use logp but we include it to make it easy to
        # swap in another policy updater that does require it, e.g. ppo-clip
        cache.add(s, a, r, done, logp)

        # update
        while cache:
            transition_batch = cache.pop()
            Gn = transition_batch.Rn  # 'Rn' is full return 'Gn' in MC-cache
            vanilla_pg.update(transition_batch, Adv=Gn)

        if done:
            break

        s = s_next
