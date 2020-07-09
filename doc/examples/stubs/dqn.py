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
q_targ = q.copy()
pi = coax.EpsilonGreedy(q, epsilon=0.1)


# specify how to update q-function
qlearning = coax.td_learning.QLearning(q, q_targ)


# specify how to trace the transitions
buffer = coax.ExperienceReplayBuffer(env, n=1, gamma=0.9, capacity=1000000)


for ep in range(100):
    pi.epsilon = ...  # exploration schedule
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # add transition to buffer
        buffer.add(s, a, r, done)

        # update
        transition_batch = buffer.sample(batch_size=32)
        qlearning.update(transition_batch)

        # periodically sync target model
        if ep % 10 == 0:
            q_targ.smooth_update(q, tau=1.0)

        if done:
            break

        s = s_next
