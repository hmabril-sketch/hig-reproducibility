"""Test predictions from the trained weights, without training (Sec. 5.3).

usage: python3 acdc_predict.py unet|deeplab DATA_DIR OUT_DIR WEIGHTS.pt [WEIGHTS.pt ...]
Writes test_gt.npy, test_img.npy, test_meta.tsv and, for every weights file named *_epXX.pt,
test_pred_epXX.npy, exactly as the training scripts do at the end of epoch XX.
Example: python3 acdc_predict.py unet acdc_data runs/unet ../weights/unet_ep02.pt ../weights/unet_ep05.pt ../weights/unet_ep16.pt
"""
import os
import re
import sys

import numpy as np
import torch

import acdc_unet as U


def main():
    kind, root, outdir, weights = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    Xte, Yte, mte = U.load_split(root, "test")
    U.save_test_set(outdir, Xte, Yte, mte)
    if kind == "unet":
        model = U.UNet()
    else:
        import acdc_deeplab
        model = acdc_deeplab.DeepLabPlus()
    for w in weights:
        ep = int(re.search(r"_ep(\d+)\.pt$", w).group(1))
        model.load_state_dict(torch.load(w, map_location="cpu"))
        np.save(os.path.join(outdir, f"test_pred_ep{ep:02d}.npy"), U.predict(model, Xte))
        print(os.path.basename(w), "->", f"test_pred_ep{ep:02d}.npy", flush=True)


if __name__ == "__main__":
    main()
