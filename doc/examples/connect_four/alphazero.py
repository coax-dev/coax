# import jax
import jax.nn
import haiku as hk
import coax
from coax.core.policy_objectives import cross_entropy_objective


class ConnectFourFuncApprox(coax.FuncApprox):

    def body(self, S, is_training):

        layers = [
            hk.Conv2D(output_channels=20, kernel_shape=4, stride=1),
            jax.nn.relu,
            hk.Conv2D(output_channels=40, kernel_shape=2, stride=1),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(64),
        ]

        try:
            # extract the action mask
            action_mask = S[:, 0, :, 0].astype('bool')
        except Exception:
            print('from inside body:', S.shape)
            raise

        # extract the state
        X_s = S[:, 1:, :, :].astype('float32')

        # forward pass
        for layer in layers:
            X_s = layer(X_s)

        X = X_s, action_mask
        return X

    def head_v(self, X):
        X_s, _ = X
        return super().head_v(X_s)

    def state_action_combiner(self, X, X_a):
        X_s, _ = X
        return super().state_action_combiner(X_s, X_a)

    def head_q2(self, X):
        X_s, _ = X
        return super().head_q2(X_s)

    def head_pi(self, X):
        X_s, action_mask = X
        P = super().head_pi(X_s).to_mapping()
        P['action_mask'] = action_mask
        return coax.Params.from_mapping(P)


env = coax.envs.ConnectFourEnv()
env = coax.wrappers.TrainMonitor(env, 'data/tensorboard')

# show logs from TrainMonitor
coax.enable_logging()


# function approximators
func = ConnectFourFuncApprox(env, learning_rate=0.001)
pi = coax.Policy(func, policy_objective=cross_entropy_objective)
v = coax.V(func, gamma=0.99, n=10, bootstrap_with_params_copy=True)
ac = coax.TDActorCritic(pi, v)
tracer = coax.caching.MonteCarlo(gamma=1)


# state_id = '20400000000000000099'
# state_id = '2020000d2c2a86ce6400'
# state_id = '10600000000000005609'  # attack
state_id = '20600000000000004d7e'  # defend
# state_id = '106000000001a021e87f'
n = coax.planning.MCTSNode(ac, state_id=state_id, random_seed=7)
# n = coax.planning.MCTSNode(ac, random_seed=17, c_puct=3.5)

n.env.render()

# n.search(n=28)
# n.show(2)
# s, pi, r, done = n.play(tau=0)
# # tracer.add(s, pi, r, done)
# n.env.render()


# for ep in range(1000):
#     n.reset()

#     for t in range(env.max_time_steps):
#         n.search(n=28)
#         # n.show(2)
#         s, pi, r, done = n.play(tau=0.)
#         tracer.add(s, pi, r, done)
#         n.env.render()

#         if done:
#             G = jnp.expand_dims(r, axis=0)
#             while tracer:
#                 transition = tracer.pop()
#                 transition.Rn = G
#                 ac.batch_update(transition)
#                 G = -G  # flip sign for opponent

#             break
