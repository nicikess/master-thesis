from typing import Union, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from torch import nn

class Metric(ABC):
    NAME = None


    @abstractmethod
    def reset_epoch_train_metrics(self):
        pass

    @abstractmethod
    def log_batch_train_metrics(self, loss, output, label, progress):
        pass

    @abstractmethod
    def log_epoch_train_metrics(self):
        pass

    @abstractmethod
    def reset_epoch_validation_metrics(self):
        pass

    @abstractmethod
    def log_batch_validation_metrics(self, output, label):
        pass

    @abstractmethod
    def log_epoch_validation_metrics(self):
        pass
