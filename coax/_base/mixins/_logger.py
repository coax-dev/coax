import logging


class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)
