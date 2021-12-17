from .q import Q


__all__ = (
    'RewardFunction',
)


class RewardFunction(Q):
    r"""

    A deterministic reward function :math:`r_\theta(s,a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass. The function signature must be the
        same as the example below.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations in a form that is convenient for
        feeding into :code:`func`. If left unspecified, this defaults to
        :func:`default_preprocessor(env.observation_space) <coax.utils.default_preprocessor>`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions in a form that is convenient for feeding into
        :code:`func`. If left unspecified, this defaults
        :func:`default_preprocessor(env.action_space) <coax.utils.default_preprocessor>`.

    value_transform : ValueTransform or pair of funcs, optional

        If provided, the target for the underlying function approximator is transformed such that:

        .. math::

            \tilde{q}_\theta(S_t, A_t)\ \approx\ f(G_t)

        This means that calling the function involves undoing this transformation:

        .. math::

            q(s, a)\ =\ f^{-1}(\tilde{q}_\theta(s, a))

        Here, :math:`f` and :math:`f^{-1}` are given by ``value_transform.transform_func`` and
        ``value_transform.inverse_func``, respectively. Note that a ValueTransform is just a
        glorified pair of functions, i.e. passing ``value_transform=(func, inverse_func)`` works
        just as well.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
