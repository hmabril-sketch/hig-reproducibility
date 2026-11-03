"""Fig. 6: two ACDC test slices whose U-Net prediction has a high Dice coefficient but a different
HIG from the reference: slice 266 (cleaned prediction, epoch 16) and slice 865 (epoch 16).
The circles are at the localized differences discussed in the text.

usage: python3 sec53_fig6.py UNET_RUN UNET_CLEAN_RUN OUT.png
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import topoqc
RUN, RUNC, OUTPNG = sys.argv[1:4]
I = np.load(os.path.join(RUN, "test_img.npy"), mmap_mode="r"); G = np.load(os.path.join(RUN, "test_gt.npy"))
P = np.load(os.path.join(RUN, "test_pred_ep16.npy")); PC = np.load(os.path.join(RUNC, "test_pred_ep16.npy"))
cm = ListedColormap(["#111111", "#e74c3c", "#f1c40f", "#3498db"])
W, H = 106, 126
def crop_for(g):
    ys, xs = np.nonzero(g > 0); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    y0 = int(np.clip(round(cy - H / 2), 0, 192 - H)); x0 = int(np.clip(round(cx - W / 2), 0, 192 - W)); return x0, y0
rows = [
  dict(i=266, pred=PC[266], title="U-Net, cleaned", circ_ref=[(70.5, 111.5)], circ_pred=[(76.1, 105.2)], crop=None),
  dict(i=865, pred=P[865], title="U-Net", circ_ref=[(81.0, 133.0)], circ_pred=[], crop=(52, 15)),
]
fig, ax = plt.subplots(2, 3, figsize=(7.9, 5.6))
letters = "abcdef"
for r, R in enumerate(rows):
    i = R["i"]; g = G[i].astype(int); p = np.asarray(R["pred"]).astype(int)
    x0, y0 = R["crop"] or crop_for(g)
    d = topoqc.dice_macro(g, p)
    sl = (slice(y0, y0 + H), slice(x0, x0 + W))
    ax[r, 0].imshow(np.asarray(I[i])[sl], cmap="gray", interpolation="nearest")
    ax[r, 1].imshow(g[sl], cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    ax[r, 2].imshow(p[sl], cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    for a, pts in ((ax[r, 1], R["circ_ref"]), (ax[r, 2], R["circ_pred"])):
        for (cy, cx) in pts:
            a.plot(cx - x0, cy - y0, "o", ms=13, mfc="none", mec="white", mew=2.2)
    titles = ["MRI slice", "reference", f"{R['title']} (Dice {d:.3f})"]
    for c in range(3):
        a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
        a.set_title(f"({letters[3 * r + c]}) {titles[c]}", fontsize=10)
    print(i, "crop", x0, y0, "dice", round(d, 3))
plt.tight_layout(); plt.savefig(OUTPNG, dpi=200); print("saved")
