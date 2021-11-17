from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

from .._base.errors import UnavailableActionError, EpisodeDoneError


__all__ = (
    'ConnectFourEnv',
)


class ConnectFourEnv(Env):
    r"""
    An adversarial environment for playing the `Connect-Four game
    <https://en.wikipedia.org/wiki/Connect_Four>`_.

    Attributes
    ----------
    action_space : gym.spaces.Discrete(7)
        The action space.

    observation_space : MultiDiscrete(nvec)

        The state observation space, representing the position of the current
        player's tokens (``s[1:,:,0]``) and the other player's tokens
        (``s[1:,:,1]``) as well as a mask over the space of actions, indicating
        which actions are available to the current player (``s[0,:,0]``) or the
        other player (``s[0,:,1]``).

        **Note:** The "current" player is relative to whose turn it is, which
        means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap between
        turns.

    max_time_steps : int
        Maximum number of timesteps within each episode.

    available_actions : array of int
        Array of available actions. This list shrinks when columns saturate.

    win_reward : 1.0
        The reward associated with a win.

    loss_reward : -1.0
        The reward associated with a loss.

    draw_reward : 0.0
        The reward associated with a draw.

    """  # noqa: E501
    # class attributes
    num_rows = 6
    num_cols = 7
    num_players = 2
    win_reward = 1.0
    loss_reward = -win_reward
    draw_reward = 0.0
    action_space = Discrete(num_cols)
    observation_space = MultiDiscrete(
        nvec=np.full((num_rows + 1, num_cols, num_players), 2, dtype='uint8'))
    max_time_steps = int(num_rows * num_cols)
    filters = np.array([
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 1, 1]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0]],
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        [[0, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0]],
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0]],
        [[0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0]],
        [[0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0]],
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1]],
    ], dtype='uint8')

    def __init__(self):
        self._init_state()

    def reset(self):
        r"""
        Reset the environment to the starting position.

        Returns
        -------
        s : 3d-array, shape: [num_rows + 1, num_cols, num_players]

            A state observation, representing the position of the current
            player's tokens (``s[1:,:,0]``) and the other player's tokens
            (``s[1:,:,1]``) as well as a mask over the space of actions,
            indicating which actions are available to the current player
            (``s[0,:,0]``) or the other player (``s[0,:,1]``).

            **Note:** The "current" player is relative to whose turn it is,
            which means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap
            between turns.

        """
        self._init_state()
        return self.state

    def step(self, a):
        r"""
        Take one step in the MDP, following the single-player convention from
        gym.

        Parameters
        ----------
        a : int, options: {0, 1, 2, 3, 4, 5, 6}
            The action to be taken. The action is the zero-based count of the
            possible insertion slots, starting from the left of the board.

        Returns
        -------
        s_next : array, shape [6, 7, 2]

            A next-state observation, representing the position of the current
            player's tokens (``s[1:,:,0]``) and the other player's tokens
            (``s[1:,:,1]``) as well as a mask over the space of actions,
            indicating which actions are available to the current player
            (``s[0,:,0]``) or the other player (``s[0,:,1]``).

            **Note:** The "current" player is relative to whose turn it is,
            which means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap
            between turns.

        r : float
            Reward associated with the transition
            :math:`(s, a)\to s_\text{next}`.

            **Note:** Since "current" player is relative to whose turn it is,
            you need to be careful about aligning the rewards with the correct
            state or state-action pair. In particular, this reward :math:`r` is
            the one associated with the :math:`s` and :math:`a`, i.e. *not*
            aligned with :math:`s_\text{next}`.

        done : bool
            Whether the episode is done.

        info : dict or None
            A dict with some extra information (or None).

        """
        if self.done:
            raise EpisodeDoneError("please reset env to start new episode")
        if not self.action_space.contains(a):
            raise ValueError(f"invalid action: {repr(a)}")
        if a not in self.available_actions:
            raise UnavailableActionError("action is not available")

        # swap players
        self._players = np.roll(self._players, -1)

        # update state
        self._state[self._levels[a], a] = self._players[0]
        self._prev_action = a

        # run logic
        self.done, reward = self._done_reward(a)
        return self.state, reward, self.done, {'state_id': self.state_id}

    def render(self, *args, **kwargs):
        r"""
        Render the current state of the environment.

        """
        # lookup for symbols
        symbol = {
            1: u'\u25CF',   # player 1 token (agent)
            2: u'\u25CB',   # player 2 token (adversary)
            -1: u'\u25BD',  # indicator for player 1's last action
            -2: u'\u25BC',  # indicator for player 2's last action
        }

        # render board
        hrule = '+---' * self.num_cols + '+\n'
        board = "  "
        board += "   ".join(
            symbol.get(-(a == self._prev_action) * self._players[1], " ")
            for a in range(self.num_cols))
        board += "  \n"
        board += hrule
        for i in range(self.num_rows):
            board += "| "
            board += " | ".join(
                symbol.get(self._state[i, j], " ")
                for j in range(self.num_cols))
            board += " |\n"
            board += hrule
        board += "  0   1   2   3   4   5   6  \n"  # actions

        print(board)

    @property
    def state(self):
        stacked_layers = np.stack((
            (self._state == self._players[0]).astype('uint8'),
            (self._state == self._players[1]).astype('uint8'),
        ), axis=-1)  # shape: [num_rows, num_cols, num_players]
        available_actions_mask = np.zeros(
            (1, self.num_cols, self.num_players), dtype='uint8')
        available_actions_mask[0, self.available_actions, :] = 1
        return np.concatenate((available_actions_mask, stacked_layers), axis=0)

    @property
    def state_id(self):
        p = str(self._players[0])
        d = '1' if self.done else '0'
        if self._prev_action is None:
            a = str(self.num_cols)
        else:
            a = str(self._prev_action)
        s = ''.join(self._state.ravel().astype('str'))  # base-3 string
        s = '{:017x}'.format(int(s, 3))  # 17-char hex string
        return p + d + a + s             # 20-char hex string

    def set_state(self, state_id):
        # decode state id
        p = int(state_id[0], 16)
        d = int(state_id[1], 16)
        a = int(state_id[2], 16)
        assert p in (1, 2)
        assert d in (0, 1)
        assert self.action_space.contains(a) or a == self.num_cols
        self._players[0] = p    # 1 or 2
        self._players[1] = 3 - p  # 2 or 1
        self.done = d == 1
        self._prev_action = None if a == self.num_cols else a
        s = np._base_repr(int(state_id[3:], 16), 3)
        z = np.zeros(self.num_rows * self.num_cols, dtype='uint8')
        z[-len(s):] = np.array(list(s), dtype='uint8')
        self._state = z.reshape((self.num_rows, self.num_cols))
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='uint8')
        for j in range(self.num_cols):
            for i in self._state[::-1, j]:
                if i == 0:
                    break
                self._levels[j] -= 1

    @property
    def available_actions(self):
        actions = np.argwhere(
            (self._levels >= 0) & (self._levels < self.num_rows)).ravel()
        assert actions.size <= self.num_cols
        return actions

    @property
    def available_actions_mask(self):
        mask = np.zeros(self.num_cols, dtype='bool')
        mask[self.available_actions] = True
        return mask

    def _init_state(self):
        self._prev_action = None
        self._players = np.array([1, 2], dtype='uint8')
        self._state = np.zeros((self.num_rows, self.num_cols), dtype='uint8')
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='uint8')
        self.done = False

    def _done_reward(self, a):
        r"""
        Check whether the last action `a` by the current player resulted in a
        win or draw for player 1 (the agent). This contains the main logic and
        implements the rules of the game.

        """
        assert self.action_space.contains(a)

        # update filling levels
        self._levels[a] -= 1

        s = self._state == self._players[0]
        for i0 in range(2, -1, -1):
            i1 = i0 + 4
            for j0 in range(4):
                j1 = j0 + 4
                if np.any(np.tensordot(self.filters, s[i0:i1, j0:j1]) == 4):
                    return True, 1.0

        # check for a draw
        if len(self.available_actions) == 0:
            return True, 0.0

        # this is what's returned throughout the episode
        return False, 0.0
