"""Serialization utilities for task distribution."""

import pickle
from typing import Any


def is_picklable(obj: Any) -> bool:
    """Check if an object can be pickled.

    Args:
        obj: Object to test

    Returns:
        True if picklable
    """
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False
