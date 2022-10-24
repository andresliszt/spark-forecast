# -*- coding: utf-8 -*-
"""Model's utilities."""


from functools import wraps
from typing import Callable


def __validator_nullifier(validator_function: Callable) -> Callable:
    """Anulates a validator."""

    @wraps(validator_function)
    # pylint: disable=unused-argument
    def decorator(*args, **kwargs):
        pass

    return decorator


def __validator_noops(validator_function: Callable) -> Callable:
    """Do nothing, if."""

    @wraps(validator_function)
    def decorator(*args, **kwargs):
        validator_function(*args, **kwargs)

    return decorator
