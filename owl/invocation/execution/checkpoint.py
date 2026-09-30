"""Checkpoint saving configuration.

This module defines the execution-level configuration used to control whether
training checkpoints are saved automatically.

Checkpoint loading and checkpoint saving belong to different invocation
domains.

``CheckpointLoad`` belongs to the component domain because it describes how
previously saved snapshot is restored into newly constructed model, optimizer, and
scheduler components.

``CheckpointSave`` belongs to the execution domain because it describes output
behavior that occurs while a training invocation is running.

Checkpoint destination directories are resolved by the workspace layer rather
than this configuration object. Keeping storage layout outside the declarative
checkpoint-saving policy avoids binding checkpoint configuration to a concrete
experiment-directory structure.

Automatic saving is disabled by default. When enabled, the training runtime
saves one checkpoint after every completed epoch.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointSave:
    """Configuration describing checkpoint output behavior.

    This object records checkpoint-saving preferences only. It does not create
    directories, generate filenames, serialize snapshot, or write files.

    The workspace layer resolves the destination directory before checkpoint
    files are written.

    Attributes:
        autosave:
            Whether the runtime should save a checkpoint automatically after
            every completed training epoch.

            When ``False``, epoch-level automatic saving is disabled.

            When ``True``, the runtime saves the complete training snapshot after
            each epoch. The saved snapshot is expected to include the model,
            optimizer, scheduler, and completed epoch progress required for
            full-snapshot resumption.
    """

    autosave: bool = False


__all__ = [
    "CheckpointSave",
]