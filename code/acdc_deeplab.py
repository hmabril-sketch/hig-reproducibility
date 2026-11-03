"""Second network of Sec. 5.3: a DeepLabv3+-style model (residual encoder, output stride 8,
atrous spatial pyramid pooling, one low-level skip, 2.3 M parameters), trained exactly like the
U-Net of acdc_unet.py (same data split, losses, optimizer, batch size, flips, epochs).

usage: python3 acdc_deeplab.py DATA_DIR OUT_DIR
Writes test_gt.npy, test_img.npy, test_meta.tsv, and test_pred_epXX.npy and deeplab_epXX.pt
for epochs 5, 10 and 16 (about 1 h on two CPU cores)."""
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import acdc_unet as U

SEED = 20261027
EPOCHS = int(os.environ.get("HIG_EPOCHS", 16))
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(int(os.environ.get("HIG_THREADS", 2)))


class Res(nn.Module):
    def __init__(self, cin, cout, stride=1, dil=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, padding=dil, dilation=dil, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, padding=dil, dilation=dil, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sk = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sk is None else self.sk(x)))


def cbr(cin, cout, k=1, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=dil * (k // 2), dilation=dil, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DeepLabPlus(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))  # 1/2
        self.l1 = nn.Sequential(Res(32, 64, 2), Res(64, 64))              # 1/4 (low-level features)
        self.l2 = nn.Sequential(Res(64, 128, 2), Res(128, 128))           # 1/8
        self.l3 = nn.Sequential(Res(128, 160, 1, 2), Res(160, 160, 1, 4))  # 1/8, dilated
        self.aspp = nn.ModuleList([cbr(160, 96, 1)] + [cbr(160, 96, 3, d) for d in (3, 6, 9)])
        self.pool = cbr(160, 96, 1)
        self.proj = cbr(5 * 96, 128, 1)
        self.low = cbr(64, 32, 1)
        self.dec = nn.Sequential(cbr(160, 96, 3), cbr(96, 96, 3))
        self.out = nn.Conv2d(96, k, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f4 = self.l1(self.stem(x))
        f8 = self.l3(self.l2(f4))
        a = [m(f8) for m in self.aspp]
        a.append(F.interpolate(self.pool(f8.mean((2, 3), keepdim=True)), size=f8.shape[-2:], mode="bilinear", align_corners=False))
        y = self.proj(torch.cat(a, 1))
        y = F.interpolate(y, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec(torch.cat([y, self.low(f4)], 1))
        return F.interpolate(self.out(y), size=(h, w), mode="bilinear", align_corners=False)


def main():
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    Xtr, Ytr, _ = U.load_split(root, "train")
    Xte, Yte, mte = U.load_split(root, "test")
    U.save_test_set(outdir, Xte, Yte, mte)
    print("train", Xtr.shape, "test", Xte.shape, flush=True)
    model = DeepLabPlus()
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 16; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = np.random.permutation(len(Xtr)); tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; xb = Xtr[idx].copy(); yb = Ytr[idx].copy()
            if np.random.rand() < 0.5:
                xb, yb = xb[:, :, ::-1].copy(), yb[:, :, ::-1].copy()
            xb = torch.from_numpy(xb[:, None]); yb = torch.from_numpy(yb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + U.dice_loss(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        print(f"epoch {ep} loss {tot / len(perm):.4f} {time.time() - t0:.0f} s", flush=True)
        if ep in (5, 10, EPOCHS):
            np.save(os.path.join(outdir, f"test_pred_ep{ep:02d}.npy"), U.predict(model, Xte))
            torch.save(model.state_dict(), os.path.join(outdir, f"deeplab_ep{ep:02d}.pt"))


if __name__ == "__main__":
    main()
