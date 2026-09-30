"""Throttled JSON Lines writers for workspace history artifacts.

This module provides an append-only JSON Lines writer that keeps one text file
open for repeated history records.

Each writer instance owns one artifact path and one independent flush interval.
The first appended record is flushed immediately. Later records remain in the
Python text buffer until the configured interval elapses, an explicit flush is
requested, or the writer is closed.
"""

import json
from pathlib import Path
from typing import Any, TextIO

from .interval import IntervalGate


class ThrottledJsonlWriter:
    """Append JSON records with time-based automatic flushing.

    Args:
        path:
            Destination path of the JSON Lines artifact.

        interval_environment_variable:
            Environment variable containing the minimum automatic flush
            interval in milliseconds.

        default_interval_ms:
            Flush interval used when the environment variable is missing,
            malformed, zero, or negative.

    Notes:
        The destination file and its parent directories are created lazily when
        the first record is appended.

        The first append is flushed immediately because the interval gate has
        not yet recorded a successful automatic flush.

        Every writer instance maintains its own interval gate. Activity in one
        history file therefore does not trigger unnecessary flush operations
        for other history files.

        ``flush`` transfers Python-buffered text to the operating system but
        does not call ``os.fsync``.

        Explicit calls to ``flush`` do not reset the automatic flush interval.
        The interval is reset only after an automatic flush triggered by
        ``append`` completes successfully.
    """

    def __init__(
        self,
        *,
        path: Path,
        interval_environment_variable: str,
        default_interval_ms: int,
    ) -> None:
        self._path = path
        self._gate = IntervalGate(
            environment_variable=interval_environment_variable,
            default_interval_ms=default_interval_ms,
        )

        self._writer: TextIO | None = None
        self._closed = False

    def append(
        self,
        record: Any,
    ) -> None:
        """Serialize and append one complete JSON Lines record.

        Serialization is completed before the destination file is modified.
        A value that cannot be encoded as JSON therefore does not leave a
        partial record in the history artifact.

        Args:
            record:
                JSON-serializable record to append.

        Raises:
            RuntimeError:
                If the writer has already been closed.

            TypeError:
                If ``record`` contains a value unsupported by the JSON encoder.

            OSError:
                If the destination cannot be created, written, or flushed.
        """
        self._require_open()

        serialized = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        writer = self._get_writer()
        writer.write(f"{serialized}\n")

        if self._gate.is_due():
            self.flush()
            self._gate.reset()

    def flush(self) -> None:
        """Flush buffered records to the operating system.

        This method has no effect before the destination file has been opened.
        Explicit flushing does not reset the automatic flush interval.

        Raises:
            RuntimeError:
                If the writer has already been closed.

            OSError:
                If the underlying text writer cannot be flushed.
        """
        self._require_open()

        if self._writer is None:
            return

        self._writer.flush()

    def close(self) -> None:
        """Flush and close the underlying text writer.

        Closing an already closed writer has no effect.

        If flushing or closing raises an exception, cleanup still attempts to
        close the underlying file handle. The writer is then marked closed and
        the first captured exception is re-raised.

        Raises:
            OSError:
                If the underlying writer cannot be flushed or closed.
        """
        if self._closed:
            return

        writer = self._writer
        first_error: BaseException | None = None

        if writer is not None:
            try:
                writer.flush()
            except BaseException as error:
                first_error = error

            try:
                writer.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error

        self._closed = True

        if first_error is not None:
            raise first_error

    def _get_writer(self) -> TextIO:
        """Return the lazily opened append-only text writer."""
        if self._writer is None:
            self._path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._writer = self._path.open(
                mode="a",
                encoding="utf-8",
                newline="\n",
            )

        return self._writer

    def _require_open(self) -> None:
        """Verify that the writer still accepts operations."""
        if self._closed:
            raise RuntimeError("JSON Lines writer has already been closed")