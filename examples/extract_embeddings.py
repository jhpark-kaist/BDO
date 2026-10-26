"""Extract subject-level BDO representations from ROI time series.

Each input is a .npy array of shape [T, 450] (time steps x ROIs, in the order of assets/roi_labels.csv).

    python examples/extract_embeddings.py --weights weights/bdo_5m.pt --tr 0.72 --out embeddings.npz data/*.npy

The normalization statistics are fitted on the inputs unless --scaler gives a saved RobustScaler.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from bdo import BDOEncoder, RobustScaler, regular_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help=".npy files, each [T, 450]")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tr", type=float, required=True, help="repetition time in seconds")
    parser.add_argument("--scaler", default=None, help="saved RobustScaler (.npz); fitted on the inputs if omitted")
    parser.add_argument("--save_scaler", default=None)
    parser.add_argument("--out", default="embeddings.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scans = [np.load(p) for p in args.inputs]
    scaler = RobustScaler.load(args.scaler) if args.scaler else RobustScaler().fit(scans)
    if args.save_scaler:
        scaler.save(args.save_scaler)

    model = BDOEncoder.from_pretrained(args.weights).to(args.device)
    embeddings = []
    for scan in scans:
        x = torch.from_numpy(scaler.transform(scan))[None].to(args.device)
        t = torch.from_numpy(regular_times(x.shape[1], args.tr))[None].to(args.device)
        embeddings.append(model.embed_subject(x, t)[0].cpu().numpy())

    np.savez(args.out, ids=np.array([Path(p).stem for p in args.inputs]), embeddings=np.stack(embeddings))
    print(f"saved {len(embeddings)} x {embeddings[0].shape[0]} embeddings to {args.out}")


if __name__ == "__main__":
    main()
