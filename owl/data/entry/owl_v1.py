"""Owl v1 dataset entry source.

This module converts an owl v1 dataset manifest into an indexed collection of
``SampleEntry`` objects.

The original owl v1 protocol requires every manifest item to contain ``tp`` and
``gt``. An empty ``gt`` value represents the absence of a ground-truth resource
and is converted to ``None``.

This reader also accepts the optional ``label`` and ``edge`` fields introduced
by the v0.0.2 entry protocol. Missing optional fields are represented by
``None``.

Original owl v1 dataset layout:

    casia_v1/
    ├── casia_v1.json
    ├── tp/
    │   ├── tampered.jpg
    │   └── authentic.jpg
    └── gt/
        └── tampered.png

Original owl v1 manifest:

    [
        {
            "tp": "tampered.jpg",
            "gt": "tampered.png"
        },
        {
            "tp": "authentic.jpg",
            "gt": ""
        }
    ]

The corresponding entries are:

    SampleEntry(
        tp=Path("casia_v1/tp/tampered.jpg"),
        gt=Path("casia_v1/gt/tampered.png"),
        label=None,
        edge=None,
    )

    SampleEntry(
        tp=Path("casia_v1/tp/authentic.jpg"),
        gt=None,
        label=None,
        edge=None,
    )

An extended manifest may additionally provide ``label`` and ``edge``:

    [
        {
            "tp": "sample.jpg",
            "gt": "sample.png",
            "label": 1,
            "edge": "sample_edge.png"
        }
    ]

The corresponding entry is:

    SampleEntry(
        tp=Path("casia_v1/tp/sample.jpg"),
        gt=Path("casia_v1/gt/sample.png"),
        label=1,
        edge=Path("casia_v1/edge/sample_edge.png"),
    )

This source only interprets dataset facts. It does not load images, inspect
resource contents, infer labels, generate masks, or derive edge supervision.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .entry import SampleEntry


class OwlV1EntrySource:
    """Indexed sample entries loaded from an owl v1 manifest.

    The manifest is read once during initialization and converted into an
    immutable tuple of ``SampleEntry`` objects. Entry order is identical to the
    order of items in the manifest.

    Args:
        root:
            Dataset root directory. For ``Path("datasets/casia_v1")``, the
            expected manifest path is
            ``datasets/casia_v1/casia_v1.json``.

    Example:
        Given this manifest:

            [
                {
                    "tp": "tampered.jpg",
                    "gt": "tampered.png"
                },
                {
                    "tp": "authentic.jpg",
                    "gt": ""
                }
            ]

        The source returns:

            source = OwlV1EntrySource("datasets/casia_v1")

            source[0]
            # SampleEntry(
            #     tp=Path("datasets/casia_v1/tp/tampered.jpg"),
            #     gt=Path("datasets/casia_v1/gt/tampered.png"),
            #     label=None,
            #     edge=None,
            # )

            source[1]
            # SampleEntry(
            #     tp=Path("datasets/casia_v1/tp/authentic.jpg"),
            #     gt=None,
            #     label=None,
            #     edge=None,
            # )

    Raises:
        FileNotFoundError:
            If the dataset root or manifest does not exist.
        NotADirectoryError:
            If the dataset root is not a directory.
        ValueError:
            If the manifest is malformed or contains invalid field values.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"dataset root does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"dataset root is not a directory: {self.root}"
            )

        self.manifest_path = self.root / f"{self.root.name}.json"
        self._entries = self._load_entries()

    def __len__(self) -> int:
        """Return the number of sample entries."""
        return len(self._entries)

    def __getitem__(self, index: int) -> SampleEntry:
        """Return the sample entry at the given index."""
        return self._entries[index]

    def _load_entries(self) -> tuple[SampleEntry, ...]:
        """Read and convert all items from the owl v1 manifest."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"owl v1 manifest does not exist: {self.manifest_path}"
            )

        try:
            with self.manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON manifest: {self.manifest_path}"
            ) from error

        if not isinstance(manifest, list):
            raise ValueError(
                "owl v1 manifest must contain a JSON list, "
                f"got {type(manifest).__name__}"
            )

        return tuple(
            self._parse_entry(item=item, index=index)
            for index, item in enumerate(manifest)
        )

    def _parse_entry(
        self,
        *,
        item: Any,
        index: int,
    ) -> SampleEntry:
        """Convert one manifest item into a ``SampleEntry``."""
        if not isinstance(item, Mapping):
            raise ValueError(
                f"manifest item {index} must be a JSON object, "
                f"got {type(item).__name__}"
            )

        if "tp" not in item:
            raise ValueError(
                f"manifest item {index} is missing required field 'tp'"
            )

        if "gt" not in item:
            raise ValueError(
                f"manifest item {index} is missing required field 'gt'"
            )

        tp = self._parse_path(
            value=item["tp"],
            field="tp",
            index=index,
            directory="tp",
            allow_empty=False,
        )
        gt = self._parse_path(
            value=item["gt"],
            field="gt",
            index=index,
            directory="gt",
            allow_empty=True,
        )
        label = self._parse_label(
            value=item.get("label"),
            index=index,
        )
        edge = self._parse_path(
            value=item.get("edge"),
            field="edge",
            index=index,
            directory="edge",
            allow_empty=True,
        )

        if tp is None:
            raise RuntimeError(
                "internal error: required field 'tp' resolved to None"
            )

        return SampleEntry(
            tp=tp,
            gt=gt,
            label=label,
            edge=edge,
        )

    def _parse_path(
        self,
        *,
        value: Any,
        field: str,
        index: int,
        directory: str,
        allow_empty: bool,
    ) -> Path | None:
        """Resolve one manifest field against its resource directory.

        For example:

            {"tp": "sample.jpg"}

        becomes:

            root / "tp" / "sample.jpg"

        When ``allow_empty`` is true, both an empty string and ``null`` are
        converted to ``None``.
        """
        if value is None or value == "":
            if allow_empty:
                return None

            raise ValueError(
                f"manifest item {index} field {field!r} must be a "
                "non-empty string"
            )

        if not isinstance(value, str):
            raise ValueError(
                f"manifest item {index} field {field!r} must be a string"
            )

        relative_path = Path(value)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"manifest item {index} field {field!r} must be a safe "
                f"relative path, got {value!r}"
            )

        return self.root / directory / relative_path

    @staticmethod
    def _parse_label(
        *,
        value: Any,
        index: int,
    ) -> int | None:
        """Validate an optional dataset-defined class index."""
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"manifest item {index} field 'label' must be an integer "
                "or null"
            )

        return value
