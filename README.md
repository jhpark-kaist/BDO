# BDO: Stochastic Optimal Control for Continuous-Time fMRI Representation Learning

Pre-trained encoders from [*Stochastic Optimal Control for Continuous-Time fMRI Representation Learning*](https://openreview.net/forum?id=N51nP3TBwR) (ICLR 2026).

This repository contains inference code only: the encoder, the input normalization, and the pre-trained weights. The pre-training code is not released yet.

## Installation

```bash
git clone https://github.com/jhpark-kaist/BDO.git && cd BDO
pip install -e .        # torch >= 2.0, numpy
```

## Pre-trained weights

| model | parameters | embedding dim | file |
|---|---|---|---|
| BDO 5M | 5.4M | 192 | [`weights/bdo_5m.pt`](weights/bdo_5m.pt) (22 MB, included) |

Only the 5M model is released for now. It is BDO (5M) in Fig. 1 and in the model-scaling experiments of the paper. The main result tables of the paper report BDO 86M, which is not released yet.

The model was pre-trained for 200 epochs on 41,072 resting-state scans from the UK Biobank. The file holds its target encoder.

## Input

Each scan is a `[T, 450]` array of mean BOLD signal per region of interest (ROI). There are 450 ROIs:
- 50 subcortical ROIs from the Tian Scale III atlas;
- 400 cortical ROIs from the Schaefer-400 (7 networks) atlas.

The columns must follow the order in [`assets/roi_labels.csv`](assets/roi_labels.csv).

Normalize the scans with `bdo.RobustScaler` (paper, App. D.1):
1. Drop the first 10 time steps.
2. Center each ROI per scan.
3. Robust-scale each ROI with the median and interquartile range over a reference set of scans.

The encoder takes the acquisition time of every step in seconds, so the sampling grid can be irregular. Scans with different repetition times (TR) can be fed directly.

## Usage

```python
import numpy as np
import torch
from bdo import BDOEncoder, RobustScaler, regular_times

model = BDOEncoder.from_pretrained("weights/bdo_5m.pt")

scans = [np.load(p) for p in paths]                # each [T, 450]
scaler = RobustScaler().fit(scans)                  # or RobustScaler.load("scaler.npz")

x = torch.from_numpy(scaler.transform(scans[0]))[None]      # [1, T, 450]
t = torch.from_numpy(regular_times(x.shape[1], tr=0.72))[None]  # [1, T], seconds

z = model(x, t)                    # [1, T, 192]  latent sequence
h = model.embed_subject(x, t)      # [1, 192]     time-averaged representation
```

Irregular sampling: pass the actual time stamps instead of `regular_times`, for example `t = torch.tensor(stamps)[None]`.

Sequences in one batch must have the same length. Scans of different lengths are encoded one at a time, or cropped to a common length.

From the command line:

```bash
python examples/extract_embeddings.py --weights weights/bdo_5m.pt --tr 0.72 --out embeddings.npz data/*.npy
```

## Citation

```bibtex
@inproceedings{
park2026stochastic,
title={Stochastic Optimal Control for Continuous-Time f{MRI} Representation Learning},
author={Joonhyeong Park and Byoungwoo Park and Chang-Bae Bang and Jungwon Choi and Hyungjin Chung and Byung-Hoon Kim and Juho Lee},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=N51nP3TBwR}
}
```
