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
v = coax.V(func)
pi = coax.Policy(func)
pi_behavior = pi.copy()


# specify how to update policy and value function
ppo_clip = coax.policy_objectives.PPOClip(pi)
simple_td = coax.td_learning.SimpleTD(v)


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


for ep in range(100):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_behavior(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # update
        if len(buffer) == buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):  # ~4 passes
                transition_batch = buffer.sample(batch_size=32)
                td_error = simple_td.td_error(transition_batch)
                ppo_clip.update(transition_batch, Adv=td_error)
                simple_td.update(transition_batch)

            buffer.clear()
            pi_behavior.soft_update(pi, tau=0.1)

        if done:
            break

        s = s_next
