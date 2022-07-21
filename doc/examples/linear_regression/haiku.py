import jax
import jax.numpy as jnp
import haiku as hk
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# create our dataset
X, y = make_regression(n_features=3)
X, X_test, y, y_test = train_test_split(X, y)


# params are defined *implicitly* in haiku
def forward(X):
    lin = hk.Linear(1)
    return lin(X).ravel()


# a transformed haiku function consists of an 'init' and an 'apply' function
forward = hk.transform(forward, apply_rng=False)

# initialize parameters
rng = jax.random.PRNGKey(seed=13)
params = forward.init(rng, X)

# redefine 'forward' as the 'apply' function
forward = forward.apply


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)


def update(params, grads):
    return jax.tree_map(lambda p, g: p - 0.05 * g, params, grads)


# the main training loop
for _ in range(50):
    loss = loss_fn(params, X_test, y_test)
    print(loss)

    grads = grad_fn(params, X, y)
    params = update(params, grads)
