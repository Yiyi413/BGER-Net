"""Atomic JSON writers for workspace artifacts.

This module provides writers for JSON artifacts whose complete contents are
replaced instead of appended.

``_AtomicJsonWriter`` publishes every value immediately through a temporary
file and atomic replacement.

``_ThrottledAtomicJsonWriter`` retains only the latest pending JSON document
and limits how frequently that document is published. Intermediate pending
values may be replaced before reaching the filesystem, which is appropriate
for workspace snapshot artifacts.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .interval import IntervalGate


class AtomicJsonWriter:
    """Atomically replace one JSON artifact.

    Args:
        path:
            Destination path of the JSON artifact.

    Notes:
        Each write is first completed in a temporary file located in the same
        directory as the destination. The temporary file is closed before
        ``os.replace`` atomically replaces the destination.

        Keeping the temporary and destination files in the same directory
        avoids crossing filesystem boundaries and preserves the atomic
        replacement semantics provided by the operating system.
    """

    def __init__(
        self,
        *,
        path: Path,
    ) -> None:
        self._path = path

    def write(
        self,
        value: Any,
    ) -> None:
        """Serialize and atomically publish a JSON document.

        Args:
            value:
                JSON-serializable value to publish.

        Raises:
            TypeError:
                If ``value`` contains a value unsupported by the JSON encoder.

            OSError:
                If the temporary file cannot be created, written, closed, or
                moved into place.
        """
        serialized = _serialize_json(value)
        self._write_serialized(serialized)

    def _write_serialized(
        self,
        serialized: str,
    ) -> None:
        """Atomically publish an already serialized JSON document."""
        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                file.write(serialized)
                file.write("\n")

            os.replace(
                temporary_path,
                self._path,
            )
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            raise


class ThrottledAtomicJsonWriter:
    """Atomically publish the latest JSON value using time-based throttling.

    Args:
        path:
            Destination path of the JSON artifact.

        interval_environment_variable:
            Environment variable containing the minimum publication interval
            in milliseconds.

        default_interval_ms:
            Publication interval used when the environment variable is
            missing, malformed, zero, or negative.

    Notes:
        The first update is published immediately.

        Later updates replace the pending in-memory document. When several
        updates arrive before the interval elapses, only the latest document is
        eventually published.

        ``flush`` bypasses the interval and publishes the latest dirty document
        immediately.

        Values are serialized during ``update`` rather than publication. This
        prevents later mutation of a caller-owned dictionary or list from
        changing the pending snapshot.
    """

    def __init__(
        self,
        *,
        path: Path,
        interval_environment_variable: str,
        default_interval_ms: int,
    ) -> None:
        self._writer = AtomicJsonWriter(
            path=path,
        )
        self._gate = IntervalGate(
            environment_variable=interval_environment_variable,
            default_interval_ms=default_interval_ms,
        )

        self._pending: str | None = None
        self._dirty = False

    def update(
        self,
        value: Any,
    ) -> None:
        """Replace the pending JSON document and publish it when due.

        Args:
            value:
                Latest JSON-serializable value represented by the artifact.

        Raises:
            TypeError:
                If ``value`` contains a value unsupported by the JSON encoder.

            OSError:
                If publication is due but the document cannot be written.
        """
        self._pending = _serialize_json(value)
        self._dirty = True

        if self._gate.is_due():
            self._publish()

    def flush(self) -> None:
        """Publish the latest dirty document regardless of the interval.

        This method has no effect when no update has been received or when the
        latest document has already been published.
        """
        if self._dirty:
            self._publish()

    def _publish(self) -> None:
        """Publish the pending document and reset the interval gate."""
        if self._pending is None:
            return

        self._writer._write_serialized(self._pending)

        self._dirty = False
        self._gate.reset()


def _serialize_json(value: Any) -> str:
    """Serialize a value into the compact workspace JSON representation.

    Serialization is completed before any destination or temporary file is
    modified. An unsupported value therefore cannot leave a partial artifact
    behind.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
