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
q = coax.Q(func)
pi = coax.EpsilonGreedy(q, epsilon=0.1)


# specify how to update q-function
qlearning = coax.td_learning.QLearning(q)


# specify how to trace the transitions
cache = coax.NStepCache(env, n=1, gamma=0.9)


for ep in range(100):
    pi.epsilon = ...  # exploration schedule
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # add transition to cache
        cache.add(s, a, r, done)

        # update
        while cache:
            transition_batch = cache.pop()
            qlearning.update(transition_batch)

        if done:
            break

        s = s_next
