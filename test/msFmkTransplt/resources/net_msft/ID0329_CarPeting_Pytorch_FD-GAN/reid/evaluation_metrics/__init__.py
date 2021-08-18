from __future__ import absolute_import

from .classification import accuracy
from .ranking import cmc, mean_ap
import ascend_function

__all__ = [
    'accuracy',
    'cmc',
    'mean_ap',
]
