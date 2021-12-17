
class CoaxError(Exception):
    pass


class SpaceError(CoaxError):
    pass


class ActionSpaceError(SpaceError):
    pass


class DistributionError(CoaxError):
    pass


class EpisodeDoneError(CoaxError):
    pass


class InconsistentCacheInputError(CoaxError):
    pass


class InsufficientCacheError(CoaxError):
    pass


class LeafNodeError(CoaxError):
    pass


class MissingAdversaryError(CoaxError):
    pass


class MissingModelError(CoaxError):
    pass


class NotLeafNodeError(CoaxError):
    pass


class NumpyArrayCheckError(CoaxError):
    pass


class TensorCheckError(CoaxError):
    pass


class UnavailableActionError(CoaxError):
    pass
