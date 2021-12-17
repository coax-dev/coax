from .stochastic_q import StochasticQ


__all__ = (
    'StochasticRewardFunction',
)


class StochasticRewardFunction(StochasticQ):
    r"""

    A stochastic reward function :math:`p_\theta(r|s,a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    value_range : tuple of floats, optional

        A pair of floats :code:`(min_value, max_value)`. If left unspecifed, this defaults to
        :code:`env.reward_range`.

    num_bins : int, optional

        The space of rewards is discretized in :code:`num_bins` equal sized bins. We use the default
        setting of 51 as suggested in the `Distributional RL <https://arxiv.org/abs/1707.06887>`_
        paper.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations in a form that is convenient for
        feeding into :code:`func`. If left unspecified, this defaults to
        :func:`default_preprocessor(env.observation_space) <coax.utils.default_preprocessor>`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions in a form that is convenient for feeding into
        :code:`func`. If left unspecified, this defaults
        :func:`default_preprocessor(env.action_space) <coax.utils.default_preprocessor>`.

    value_transform : ValueTransform or pair of funcs, optional

        If provided, the target for the underlying function approximator is transformed:

        .. math::

            \tilde{G}_t\ =\ f(G_t)

        This means that calling the function involves undoing this transformation using its inverse
        :math:`f^{-1}`. The functions :math:`f` and :math:`f^{-1}` are given by
        ``value_transform.transform_func`` and ``value_transform.inverse_func``, respectively. Note
        that a ValueTransform is just a glorified pair of functions, i.e. passing
        ``value_transform=(func, inverse_func)`` works just as well.

    random_seed : int, optional

        Seed for pseudo-random number generators.


    """
    def __init__(
            self, func, env, value_range=None, num_bins=51, observation_preprocessor=None,
            action_preprocessor=None, value_transform=None, random_seed=None):

        super().__init__(
            func, env, value_range=(value_range or env.reward_range), num_bins=51,
            observation_preprocessor=None, action_preprocessor=None, value_transform=None,
            random_seed=None)
