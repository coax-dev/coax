from abc import ABC, abstractmethod


__all__ = (
    'BaseReplayBuffer',
)


class BaseReplayBuffer(ABC):

    @property
    @abstractmethod
    def capacity(self):
        pass

    @abstractmethod
    def add(self, transition_batch):
        pass

    @abstractmethod
    def sample(self, batch_size=32):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
