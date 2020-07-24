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
q = coax.Q(func)


# target networks
pi_targ = pi.copy()
q_targ = q.copy()


# specify how to update policy and value function
determ_pg = coax.policy_objectives.DeterministicPG(pi, q)
qlearning = coax.td_learning.QLearningMode(q, pi_targ, q_targ)  # or use Sarsa


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


for ep in range(100):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # update
        transition_batch = buffer.sample(batch_size=32)
        determ_pg.update(transition_batch)
        qlearning.update(transition_batch)

        # periodically sync target models
        if ep % 10 == 0:
            pi_targ.soft_update(pi, tau=1.0)
            q_targ.soft_update(q, tau=1.0)

        if done:
            break

        s = s_next
