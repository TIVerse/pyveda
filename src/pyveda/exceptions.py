"""Exception hierarchy for PyVeda."""


class VedaError(Exception):
    """Base exception for all PyVeda errors."""

    pass


class SchedulerError(VedaError):
    """Raised when scheduler encounters an error."""

    pass


class ExecutorError(VedaError):
    """Raised when executor fails to execute a task."""

    pass


class GPUError(VedaError):
    """Raised when GPU operations fail."""

    pass


class ConfigurationError(VedaError):
    """Raised when configuration is invalid."""

    pass


class TaskError(VedaError):
    """Raised when task execution fails."""

    pass


class TimeoutError(VedaError):
    """Raised when an operation times out."""

    pass
