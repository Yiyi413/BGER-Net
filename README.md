# BGER-Net

Code for **BGER-Net: A Lightweight Boundary-Guided Evidence Routing Network for Image Manipulation Localization**.

## Requirements

Use a Python environment with PyTorch and CUDA support for GPU training. Install PyTorch and torchvision for your CUDA version using the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
pip install timm albumentations numpy pillow matplotlib
```

The `owl` training framework is included in this repository; it does not need a separate installation.

## Data preparation

Place the three tampCOCO training subsets and the four evaluation datasets under `data/` as follows:

```text
data/
  bcm_coco30000/
  cm_coco30000/
  sp_coco30000/
  nist16/
  columbia/
  coverage/
  casia_v1/
```

Each dataset directory must contain a JSON manifest named after the directory, an image folder `tp/`, and a mask folder `gt/`. For example:

```text
data/casia_v1/
  casia_v1.json
  tp/sample.jpg
  gt/sample.png
```

The manifest is a JSON list of image/mask pairs:

```json
[
  {"tp": "sample.jpg", "gt": "sample.png"}
]
```

The paths in `tp` and `gt` are relative to their respective folders. Use an empty `gt` string for an authentic image without a mask. The loader converts masks to binary values. An optional `edge` field can point to a file in `edge/`; when it is omitted, boundary targets are derived from the ground-truth masks during training. Dataset files and pretrained weights are not included in this repository.

## Training and cross-dataset evaluation

Place the pretrained ConvNeXt-Tiny checkpoint at `pretrained/ConvNeXt-Tiny.pth`, then run from the repository root:

```bash
python train_bger.py
```

The default configuration trains the full model for 30 epochs with a batch size of 16, `512 x 512` inputs, AdamW (learning rate `1e-4`, weight decay `1e-2`), and a boundary-loss weight of `0.3`. After each epoch, the script evaluates pixel-level F1 and AUC on NIST16, Columbia, COVERAGE, and CASIA v1 without fine-tuning on these datasets. The evaluation threshold for F1 is `0.5`.

To use a checkpoint stored elsewhere or resume a run:

```bash
python train_bger.py --backbone-checkpoint /path/to/ConvNeXt-Tiny.pth
python train_bger.py --resume /path/to/epoch-0010.pt
```

Run `python train_bger.py --help` for all available options. Outputs are written to a timestamped directory under `runs/bger/`, including `launch_config.json`, per-epoch checkpoints in `checkpoints/`, and evaluation results in `history/evaluation.jsonl`. The evaluation datasets are assessed during training; they are not used for optimizer updates.

This entry point covers the main training run and per-epoch cross-dataset evaluation. Separate commands for the paper's ablation and robustness experiments are not included in the current public repository.
