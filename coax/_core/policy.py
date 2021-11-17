from ..utils import default_preprocessor
from ..proba_dists import ProbaDist
from .base_stochastic_func_type2 import BaseStochasticFuncType2


class Policy(BaseStochasticFuncType2):
    r"""

    A parametrized policy :math:`\pi_\theta(a|s)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations in a form that is convenient for
        feeding into :code:`func`. If left unspecified, this defaults to
        :func:`default_preprocessor(env.observation_space) <coax.utils.default_preprocessor>`.

    proba_dist : ProbaDist, optional

        A probability distribution that is used to interpret the output of :code:`func
        <coax.Policy.func>`. Check out the :mod:`coax.proba_dists` module for available options.

        If left unspecified, this defaults to:

        .. code:: python

            proba_dist = coax.proba_dists.ProbaDist(action_space)

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(self, func, env, observation_preprocessor=None, proba_dist=None, random_seed=None):

        # defaults
        if observation_preprocessor is None:
            observation_preprocessor = default_preprocessor(env.observation_space)
        if proba_dist is None:
            proba_dist = ProbaDist(env.action_space)

        super().__init__(
            func=func,
            observation_space=env.observation_space,
            action_space=env.action_space,
            observation_preprocessor=observation_preprocessor,
            proba_dist=proba_dist,
            random_seed=random_seed)

    def __call__(self, s, return_logp=False):
        r"""

        Sample an action :math:`a\sim\pi_\theta(.|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        return_logp : bool, optional

            Whether to return the log-propensity :math:`\log\pi(a|s)`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        logp : float, optional

            The log-propensity :math:`\log\pi_\theta(a|s)`. This is only returned if we set
            ``return_logp=True``.

        """
        return super().__call__(s, return_logp=return_logp)

    def mean(self, s):
        r"""

        Get the mean of the distribution :math:`\pi_\theta(.|s)`.

        Note that if the actions are discrete, this returns the :attr:`mode` instead.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        """
        return super().mean(s)

    def mode(self, s):
        r"""

        Sample a greedy action :math:`a=\arg\max_a\pi_\theta(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        """
        return super().mode(s)

    def dist_params(self, s):
        r"""

        Get the conditional distribution parameters of :math:`\pi_\theta(.|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        dist_params : Params

            The distribution parameters of :math:`\pi_\theta(.|s)`.

        """
        return super().dist_params(s)

    @classmethod
    def example_data(
            cls, env, observation_preprocessor=None, proba_dist=None,
            batch_size=1, random_seed=None):

        # defaults
        if observation_preprocessor is None:
            observation_preprocessor = default_preprocessor(env.observation_space)
        if proba_dist is None:
            proba_dist = ProbaDist(env.action_space)

        return super().example_data(
            env=env, observation_preprocessor=observation_preprocessor, proba_dist=proba_dist,
            batch_size=batch_size, random_seed=random_seed)
