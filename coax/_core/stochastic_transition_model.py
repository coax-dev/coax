from ..utils import default_preprocessor
from ..proba_dists import ProbaDist
from .base_stochastic_func_type1 import BaseStochasticFuncType1


__all__ = (
    'StochasticTransitionModel',
)


class StochasticTransitionModel(BaseStochasticFuncType1):
    r"""

    A stochastic transition model :math:`p_\theta(s'|s,a)`. Here, :math:`s'` is the successor state,
    given that we take action :math:`a` from state :math:`s`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations in a form that is convenient for
        feeding into :code:`func`. If left unspecified, this defaults to
        :attr:`proba_dist.preprocess_variate <coax.proba_dists.ProbaDist.preprocess_variate>`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions in a form that is convenient for feeding into
        :code:`func`. If left unspecified, this defaults
        :func:`default_preprocessor(env.action_space) <coax.utils.default_preprocessor>`.

    proba_dist : ProbaDist, optional

        A probability distribution that is used to interpret the output of :code:`func
        <coax.Policy.func>`. Check out the :mod:`coax.proba_dists` module for available options.

        If left unspecified, this defaults to:

        .. code:: python

            proba_dist = coax.proba_dists.ProbaDist(observation_space)

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, env, observation_preprocessor=None, action_preprocessor=None,
            proba_dist=None, random_seed=None):

        # set defaults
        if proba_dist is None:
            proba_dist = ProbaDist(env.observation_space)
        if observation_preprocessor is None:
            observation_preprocessor = proba_dist.preprocess_variate
        if action_preprocessor is None:
            action_preprocessor = default_preprocessor(env.action_space)

        super().__init__(
            func=func,
            observation_space=env.observation_space,
            action_space=env.action_space,
            observation_preprocessor=observation_preprocessor,
            action_preprocessor=action_preprocessor,
            proba_dist=proba_dist,
            random_seed=random_seed)

    @classmethod
    def example_data(
            cls, env, action_preprocessor=None, proba_dist=None, batch_size=1, random_seed=None):

        # set defaults
        if action_preprocessor is None:
            action_preprocessor = default_preprocessor(env.action_space)
        if proba_dist is None:
            proba_dist = ProbaDist(env.observation_space)

        return super().example_data(
            env=env,
            observation_preprocessor=proba_dist.preprocess_variate,
            action_preprocessor=action_preprocessor,
            proba_dist=proba_dist,
            batch_size=batch_size,
            random_seed=random_seed)

    def __call__(self, s, a=None, return_logp=False):
        r"""

        Sample a successor state :math:`s'` from the dynamics model :math:`p(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        return_logp : bool, optional

            Whether to return the log-propensity :math:`\log p(s'|s,a)`.

        Returns
        -------
        s_next : state observation or list thereof

            Depending on whether :code:`a` is provided, this either returns a single next-state
            :math:`s'` or a list of :math:`n` next-states, one for each discrete action.

        logp : non-positive float or list thereof, optional

            The log-propensity :math:`\log p(s'|s,a)`. This is only returned if we set
            ``return_logp=True``. Depending on whether :code:`a` is provided, this is either a
            single float or a list of :math:`n` floats, one for each discrete action.

        """
        return super().__call__(s, a=a, return_logp=return_logp)

    def mean(self, s, a=None):
        r"""

        Get the mean successor state :math:`s'` according to the dynamics model,
        :math:`s'=\arg\max_{s'}p_\theta(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        s_next : state observation or list thereof

            Depending on whether :code:`a` is provided, this either returns a single next-state
            :math:`s'` or a list of :math:`n` next-states, one for each discrete action.

        """
        return super().mean(s, a=a)

    def mode(self, s, a=None):
        r"""

        Get the most probable successor state :math:`s'` according to the dynamics model,
        :math:`s'=\arg\max_{s'}p_\theta(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        s_next : state observation or list thereof

            Depending on whether :code:`a` is provided, this either returns a single next-state
            :math:`s'` or a list of :math:`n` next-states, one for each discrete action.

        """
        return super().mode(s, a=a)

    def dist_params(self, s, a=None):
        r"""

        Get the parameters of the conditional probability distribution :math:`p_\theta(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        dist_params : dict or list of dicts

            Depending on whether :code:`a` is provided, this either returns a single dist-params
            dict or a list of :math:`n` such dicts, one for each discrete action.

        """
        return super().dist_params(s, a=a)
