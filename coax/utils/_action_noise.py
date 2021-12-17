import numpy as onp


__all__ = (
    'OrnsteinUhlenbeckNoise',
)


class OrnsteinUhlenbeckNoise:
    r"""

    Add `Ornstein-Uhlenbeck <https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process>`_ noise to
    continuous actions.

    .. math::

        A_t\ \mapsto\ \widetilde{A}_t = A_t + X_t

    As a side effect, the Ornstein-Uhlenbeck noise :math:`X_t` is updated with every function call:

    .. math::

        X_t\ =\ X_{t-1} - \theta\,\left(X_{t-1} - \mu\right) + \sigma\,\varepsilon

    where :math:`\varepsilon` is white noise, i.e. :math:`\varepsilon\sim\mathcal{N}(0,\mathbb{I})`.

    The authors of the `DDPG paper <https://arxiv.org/abs/1509.02971>`_ chose to use
    Ornstein-Uhlenbeck noise "*[...] in order to explore well in physical environments that have
    momentum.*"


    Parameters
    ----------
    mu : float or ndarray, optional

        The mean :math:`\mu` towards which the Ornstein-Uhlenbeck process should revert; must be
        broadcastable with the input actions.

    sigma : positive float or ndarray, optional

        The spread of the noise :math:`\sigma>0` of the Ornstein-Uhlenbeck process; must be
        broadcastable with the input actions.

    theta : positive float or ndarray, optional

        The (element-wise) dissipation rate :math:`\theta>0` of the Ornstein-Uhlenbeck process; must
        be broadcastable with the input actions.

    min_value : float or ndarray, optional

        The lower bound used for clipping the output action; must be broadcastable with the input
        actions.

    max_value : float or ndarray, optional

        The upper bound used for clipping the output action; must be broadcastable with the input
        actions.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """

    def __init__(
            self, mu=0., sigma=1., theta=0.15, min_value=None, max_value=None, random_seed=None):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.min_value = -1e15 if min_value is None else min_value
        self.max_value = 1e15 if max_value is None else max_value
        self.random_seed = random_seed
        self.rnd = onp.random.RandomState(self.random_seed)
        self.reset()

    def reset(self):
        r"""

        Reset the Ornstein-Uhlenbeck process.

        """
        self._noise = None

    def __call__(self, a):
        r"""

        Add some Ornstein-Uhlenbeck to a continuous action.

        Parameters
        ----------
        a : action

            A single action :math:`A_t`.

        Returns
        -------
        a_noisy : action

            An action with noise added :math:`\widetilde{A}_t = A_t + X_t`.

        """
        a = onp.asarray(a)
        if self._noise is None:
            self._noise = onp.ones_like(a) * self.mu

        white_noise = onp.asarray(self.rnd.randn(*a.shape), dtype=a.dtype)
        self._noise += self.theta * (self.mu - self._noise) + self.sigma * white_noise
        self._noise = onp.clip(self._noise, self.min_value, self.max_value)
        return a + self._noise
