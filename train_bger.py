from __future__ import annotations

import argparse
import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from owl import invoke
from owl.data.augment import infer as infer_transform
from owl.data.augment import train as train_transform
from owl.data.entry import OwlV1EntrySource
from owl.invocation.components.checkpoint import CheckpointLoad
from owl.invocation.components.components import TrainComponents
from owl.invocation.data.infer import InferData
from owl.invocation.data.train import TrainData
from owl.invocation.execution.checkpoint import CheckpointSave
from owl.invocation.execution.train import TrainExecution
from owl.invocation.train import TrainInference, TrainInvocation
from owl.optim import adamw, poly
from owl.processors.evaluator.binary_mask import BinaryMaskEvaluator

from models.bger_net import BGERNet, OwlBGERCriterion


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONE_CHECKPOINT = (
    PROJECT_ROOT / "pretrained" / "ConvNeXt-Tiny.pth"
)
TRAIN_ROOTS = (
    PROJECT_ROOT / "data" / "bcm_coco30000",
    PROJECT_ROOT / "data" / "cm_coco30000",
    PROJECT_ROOT / "data" / "sp_coco30000",
)
GENERALIZATION_ROOTS = {
    "NIST16": PROJECT_ROOT / "data" / "nist16",
    "Columbia": PROJECT_ROOT / "data" / "columbia",
    "COVERAGE": PROJECT_ROOT / "data" / "coverage",
    "CASIA_v1": PROJECT_ROOT / "data" / "casia_v1",
}


class LimitedOwlV1EntrySource:
    """Limit an Owl v1 source for fast end-to-end smoke tests."""

    def __init__(self, root: str | Path, limit: int):
        self.source = OwlV1EntrySource(root)
        self.limit = min(limit, len(self.source))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        if not 0 <= index < self.limit:
            raise IndexError(index)
        return self.source[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BGER-Net with Owl and test generalization every epoch"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--lambda-boundary", type=float, default=0.3)
    parser.add_argument(
        "--ablation",
        choices=("A", "B", "C", "D", "E", "F"),
        default="F",
        help="A=RGB, B=dual-add, C=+cross-attn, D=+experts, E=+single-router, F=full",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backbone-name", default="convnext_tiny")
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=DEFAULT_BACKBONE_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "bger",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-samples-per-dataset", type=int, default=0)
    parser.add_argument("--max-eval-samples-per-dataset", type=int, default=0)
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        choices=tuple(GENERALIZATION_ROOTS),
        default=list(GENERALIZATION_ROOTS),
        help="Generalization datasets evaluated after each epoch.",
    )
    return parser.parse_args()


def _loader_options(
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        options["persistent_workers"] = True
    return options


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.workers < 0 or args.eval_workers < 0:
        raise ValueError("worker counts must be non-negative")
    if (
        args.max_train_samples_per_dataset < 0
        or args.max_eval_samples_per_dataset < 0
    ):
        raise ValueError("sample limits must be non-negative")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.lambda_boundary < 0.0:
        raise ValueError("lambda_boundary must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested, but CUDA is unavailable")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.resume}")
    if (
        not args.no_pretrained
        and args.backbone_checkpoint is not None
        and not args.backbone_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"backbone checkpoint does not exist: {args.backbone_checkpoint}"
        )


def _validate_dataset(root: Path) -> None:
    manifest = root / f"{root.name}.json"
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Owl dataset manifest does not exist: {manifest}")


def validate_inputs(args: argparse.Namespace) -> None:
    _validate_args(args)
    eval_roots = (GENERALIZATION_ROOTS[name] for name in args.eval_datasets)
    for root in (*TRAIN_ROOTS, *eval_roots):
        _validate_dataset(root)


def build_invocation(
    args: argparse.Namespace,
    run_directory: Path,
) -> TrainInvocation:
    image_size = (args.image_size, args.image_size)
    pin_memory = args.device.startswith("cuda")
    backbone_checkpoint = (
        None if args.no_pretrained else args.backbone_checkpoint
    )
    train_entry_source = (
        partial(
            LimitedOwlV1EntrySource,
            limit=args.max_train_samples_per_dataset,
        )
        if args.max_train_samples_per_dataset > 0
        else OwlV1EntrySource
    )
    eval_entry_source = (
        partial(
            LimitedOwlV1EntrySource,
            limit=args.max_eval_samples_per_dataset,
        )
        if args.max_eval_samples_per_dataset > 0
        else OwlV1EntrySource
    )
    selected_generalization_roots = {
        name: GENERALIZATION_ROOTS[name] for name in args.eval_datasets
    }

    training_data = TrainData(
        sources=list(TRAIN_ROOTS),
        default_entry_source=train_entry_source,
        augment=train_transform(image_size),
        loader=_loader_options(
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
            pin_memory=pin_memory,
        ),
    )

    # These datasets are used only for fixed generalization testing. They do
    # not participate in optimization or model selection.
    generalization_data = InferData(
        sources=selected_generalization_roots,
        default_entry_source=eval_entry_source,
        augment=infer_transform(image_size),
        loader=_loader_options(
            batch_size=args.eval_batch_size,
            workers=args.eval_workers,
            shuffle=False,
            pin_memory=pin_memory,
        ),
    )

    checkpoint = (
        CheckpointLoad(path=args.resume, model_only=False)
        if args.resume is not None
        else None
    )
    components = TrainComponents(
        model=(
            BGERNet,
            {
                "backbone_name": args.backbone_name,
                "pretrained": not args.no_pretrained,
                "freeze_backbone": False,
                "backbone_checkpoint": backbone_checkpoint,
                "ablation": args.ablation,
            },
        ),
        criterion=(
            OwlBGERCriterion,
            {
                "lambda_boundary": (
                    args.lambda_boundary if args.ablation == "F" else 0.0
                )
            },
        ),
        optimizer=adamw(
            lr=args.lr,
            weight_decay=args.weight_decay,
        ),
        scheduler=poly(power=args.poly_power),
        checkpoint=checkpoint,
    )

    execution = TrainExecution(
        total_epochs=args.epochs,
        device=args.device,
        workspace=run_directory,
        checkpoint=CheckpointSave(autosave=True),
    )

    return TrainInvocation(
        components=components,
        data=training_data,
        execution=execution,
        inference=TrainInference(
            data=generalization_data,
            evaluator=(
                BinaryMaskEvaluator,
                {"threshold": args.threshold},
            ),
        ),
    )


def create_run_directory(root: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S-%f")[:-3]
    run_directory = root.resolve() / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def save_launch_config(args: argparse.Namespace, run_directory: Path) -> None:
    config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training_datasets": [str(path) for path in TRAIN_ROOTS],
        "generalization_datasets": {
            name: str(GENERALIZATION_ROOTS[name]) for name in args.eval_datasets
        },
        "generalization_metrics": ["pixel_f1", "pixel_auc"],
        "effective_lambda_boundary": (
            args.lambda_boundary if args.ablation == "F" else 0.0
        ),
        "resolved_backbone_checkpoint": (
            str(args.backbone_checkpoint) if not args.no_pretrained else None
        ),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }
    (run_directory / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    run_directory = create_run_directory(args.workspace_root)
    save_launch_config(args, run_directory)
    invocation = build_invocation(args, run_directory)

    print(f"run_directory={run_directory}")
    print("train=bcm_coco30000+cm_coco30000+sp_coco30000 (90000 samples)")
    print(f"generalization={','.join(args.eval_datasets)} after every epoch")
    print(f"device={args.device}, image={args.image_size}, batch={args.batch_size}")
    print(
        f"ablation={args.ablation}, boundary_loss="
        f"{args.lambda_boundary if args.ablation == 'F' else 0.0}"
    )
    invoke(invocation)


if __name__ == "__main__":
    main()
