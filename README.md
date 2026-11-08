# BGER-Net

This repository contains the code for our paper:

### BGER-Net: A Lightweight Boundary-Guided Evidence Routing Network for Image Manipulation Localization

Lichao Su, Yucheng Yi, Kun Zou, Shiyan Tu

![Overview of BGER-Net](model.png)

## Datasets

Training: [tampCOCO](https://www.kaggle.com/datasets/qsii24/tampcoco) (use the selected splicing and copy-move subsets).

Testing:

- [CASIA v1](https://github.com/namtpham/casia1groundtruth) (community mirror with ground-truth masks; the original server is unavailable)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) (official download page)
- [COVERAGE](https://github.com/wenbihan/coverage) (author's dataset page)
- [NIST16](https://mfc.nist.gov/) (official NIST portal; registration and a data-use agreement may be required)

Place the downloaded datasets under `data/` and set `TRAIN_ROOTS` and `GENERALIZATION_ROOTS` in `train_bger.py` to their extracted locations. Each dataset must use the Owl v1 layout: `tp/` for images, `gt/` for masks, and a `<dataset-folder-name>.json` manifest containing `tp`/`gt` filename pairs. The training pipeline uses [Owl](https://github.com/aikwen/owl), which is included in this repository.

## Training and testing

Install PyTorch for your system, then install the remaining dependencies:

```bash
pip install timm albumentations numpy pillow matplotlib
```

Place a pretrained ConvNeXt-Tiny checkpoint at `pretrained/ConvNeXt-Tiny.pth` and run:

```bash
python train_bger.py
```

The script trains BGER-Net on tampCOCO and evaluates it on the four test datasets after each epoch, without target-dataset fine-tuning. Checkpoints and F1/AUC results are saved under `runs/bger/`.
