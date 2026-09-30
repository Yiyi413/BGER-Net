"""Time interval control for workspace writers.

This module defines a lightweight interval gate shared by throttled workspace
writers. The gate determines whether enough monotonic time has elapsed since
the previous successful I/O operation.

Configured intervals are read dynamically from environment variables. This
allows code running in the current process to adjust writer behavior without
reconstructing the writer instance.
"""

import os
import time


class IntervalGate:
    """Control the minimum interval between successful I/O operations.

    The gate does not execute I/O itself. Writers call ``is_due`` before an
    optional operation and call ``reset`` only after that operation succeeds.

    Args:
        environment_variable:
            Name of the environment variable containing the interval in
            milliseconds.

        default_interval_ms:
            Interval used when the environment variable is missing, malformed,
            zero, or negative.

    Notes:
        The first ``is_due`` call returns ``True`` because no successful
        operation has been recorded yet.

        Elapsed time is measured with ``time.monotonic`` so system clock
        changes do not affect throttling behavior.

        Environment variables changed by another process are not visible to
        the current process. Dynamic reads support changes made through
        ``os.environ`` inside the running process.
    """

    def __init__(
        self,
        *,
        environment_variable: str,
        default_interval_ms: int,
    ) -> None:
        if default_interval_ms <= 0:
            raise ValueError("default interval must be greater than zero")

        self._environment_variable = environment_variable
        self._default_interval_ms = default_interval_ms
        self._last_reset_at: float | None = None

    def is_due(self) -> bool:
        """Return whether the controlled operation may run now.

        Returns:
            ``True`` when no successful operation has been recorded or when
            the configured interval has elapsed since the latest reset.
        """
        if self._last_reset_at is None:
            return True

        elapsed_seconds = time.monotonic() - self._last_reset_at
        interval_seconds = self._interval_ms() / 1000

        return elapsed_seconds >= interval_seconds

    def reset(self) -> None:
        """Record that the controlled operation completed successfully."""
        self._last_reset_at = time.monotonic()

    def _interval_ms(self) -> int:
        """Return the currently configured interval in milliseconds.

        Missing, malformed, zero, and negative environment values fall back to
        the default interval.
        """
        raw_value = os.getenv(self._environment_variable)

        if raw_value is None:
            return self._default_interval_ms

        try:
            interval = int(raw_value)
        except ValueError:
            return self._default_interval_ms

        if interval <= 0:
            return self._default_interval_ms

        return interval