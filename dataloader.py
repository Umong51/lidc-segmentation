import functools

import monai.transforms as T
import numpy as np
import pylidc as pl
import torch
from pylidc.utils import consensus
from torch.utils.data import DataLoader, Dataset


class CFG:
    num_samples = 16
    spatial_size = (64, 64, 64)
    mask_thresh = 16


class Ct:
    transforms = T.Compose(
        [
            T.FgBgToIndicesd(keys=["vol", "mask"]),
            T.RandCropByPosNegLabeld(
                keys=["vol", "mask"],
                label_key="mask",
                pos=7,
                neg=3,
                num_samples=CFG.num_samples,
                spatial_size=CFG.spatial_size,
            ),
        ]
    )

    def __init__(self, scan: pl.Scan):
        vol = scan.to_volume(verbose=False)
        vol = vol.clip(-1000, 1000)
        vol = vol / 1000

        mask = np.zeros(vol.shape, dtype=bool)
        nodules = scan.cluster_annotations()

        for nod in nodules:
            nod_mask, nod_bbox, nod_masks = consensus(nod, pad=512)
            if nod_mask.sum() < CFG.mask_thresh:
                continue
            mask |= nod_mask

        vol = vol.astype(np.float32)
        mask = mask.astype(np.float32)

        self.vol = torch.tensor(vol).unsqueeze(0)
        self.mask = torch.tensor(mask).unsqueeze(0)

    @functools.cached_property
    def samples(self):
        inp = dict(vol=self.vol, mask=self.mask)
        out = Ct.transforms(inp)
        return [(sample["vol"], sample["mask"]) for sample in out]


@functools.lru_cache(maxsize=1)
def get_ct(scan: pl.Scan):
    return Ct(scan)


class LIDCDataset(Dataset):
    def __init__(self, scans):
        self.scans = scans

    def __len__(self):
        return len(self.scans) * CFG.num_samples

    def __getitem__(self, index):
        scan_idx = index // CFG.num_samples
        sample_idx = index % CFG.num_samples

        scan = self.scans[scan_idx]
        return get_ct(scan).samples[sample_idx]


def get_loaders():
    scans = pl.query(pl.Scan).all()
    train_scans = []
    val_scans = []

    for i, scan in enumerate(scans):
        if i % 5 == 0:
            val_scans.append(scan)
        else:
            train_scans.append(scan)

    train_ds = LIDCDataset(train_scans)
    val_ds = LIDCDataset(val_scans)

    train_loader = DataLoader(train_ds, batch_size=8)
    val_loader = DataLoader(val_ds, batch_size=8)

    return train_loader, val_loader
