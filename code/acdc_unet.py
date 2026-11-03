"""Small 2D U-Net of Sec. 5.3 on ACDC short-axis slices (CPU), saving test predictions at
epochs 2, 5 and 16 so that predictions of different quality are available.

usage: python3 acdc_unet.py DATA_DIR OUT_DIR
DATA_DIR contains, in any subfolders, the files patientXXX_sax_ed.nii.gz,
patientXXX_sax_ed_gt.nii.gz, patientXXX_sax_es.nii.gz, patientXXX_sax_es_gt.nii.gz (see README);
patients 1-100 are used for training and 101-150 for testing.
Writes test_gt.npy, test_img.npy, test_meta.tsv, and test_pred_epXX.npy and unet_epXX.pt
(about 50 min on two CPU cores).
Labels: 0 background, 1 RV, 2 MYO, 3 LV.
"""
import glob
import re
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 192          # full field of view: the heart can reach the border of a 128 crop
EPOCHS = int(os.environ.get("HIG_EPOCHS", 16))
SAVE_AT = tuple(int(e) for e in os.environ.get("HIG_SAVE_AT", "2,5,16").split(","))
SEED = 20260927
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


def _pairs(root):
    """Every annotated frame (image + *_gt mask) under root, whatever the naming
    (frameXX or sax_ed / sax_es); cine series (*_t, *_4d) have no mask."""
    out = []
    for gt in sorted(glob.glob(os.path.join(root, "**", "*_gt.nii.gz"), recursive=True)):
        img = gt.replace("_gt.nii.gz", ".nii.gz")
        if os.path.exists(img):
            pid = int(re.search(r"patient(\d+)", os.path.basename(gt)).group(1))
            out.append((pid, img, gt))
    return out


def load_split(root, split):
    """ACDC convention: patients 1-100 train, 101-150 test."""
    xs, ys, meta = [], [], []
    for pid, img, gt in _pairs(root):
        if (split == "train") != (pid <= 100):
            continue
        x = nib.load(img).get_fdata().astype(np.float32)
        y = nib.load(gt).get_fdata().astype(np.int64)
        lo, hi = np.percentile(x, [1, 99])
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        h, w = x.shape[:2]
        y0, x0 = (h - CROP) // 2, (w - CROP) // 2
        for s in range(x.shape[2]):
            xs.append(x[y0:y0 + CROP, x0:x0 + CROP, s])
            ys.append(y[y0:y0 + CROP, x0:x0 + CROP, s])
            meta.append((os.path.basename(img).replace(".nii.gz", ""), s, x.shape[2]))
    return np.stack(xs), np.stack(ys), meta


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=16, k=4):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        self.out = nn.Conv2d(c, k, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def dice_loss(logits, y, k=4):
    p = logits.softmax(1)
    oh = F.one_hot(y, k).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


def predict(model, X, bs=32):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs, None])
            out.append(model(xb).argmax(1).numpy().astype(np.uint8))
    return np.concatenate(out)


def save_test_set(outdir, Xte, Yte, mte):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "test_gt.npy"), Yte.astype(np.uint8))
    np.save(os.path.join(outdir, "test_img.npy"), Xte)
    with open(os.path.join(outdir, "test_meta.tsv"), "w") as f:
        for name, s, n in mte:
            f.write(f"{name}\t{s}\t{n}\n")


def main():
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    Xtr, Ytr, _ = load_split(root, "train")
    Xte, Yte, mte = load_split(root, "test")
    print("train slices", Xtr.shape, "test slices", Xte.shape, flush=True)
    save_test_set(outdir, Xte, Yte, mte)
    model = UNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 16
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].copy()
            yb = Ytr[idx].copy()
            if np.random.rand() < 0.5:           # simple flip augmentation
                xb, yb = xb[:, :, ::-1].copy(), yb[:, :, ::-1].copy()
            xb = torch.from_numpy(xb[:, None])
            yb = torch.from_numpy(yb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        print(f"epoch {ep} loss {tot / len(perm):.4f} {time.time() - t0:.0f} s", flush=True)
        if ep in SAVE_AT:
            np.save(os.path.join(outdir, f"test_pred_ep{ep:02d}.npy"), predict(model, Xte))
            torch.save(model.state_dict(), os.path.join(outdir, f"unet_ep{ep:02d}.pt"))


if __name__ == "__main__":
    main()
