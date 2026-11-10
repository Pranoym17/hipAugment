"""DataLoader demo: the CPU pads mixed-size images into one batch, the GPU does the rest.

    PYTHONPATH=python python3 python/dataloader_demo.py
"""
import torch
from torch.utils.data import DataLoader, Dataset

import hipaugment


class RandomImages(Dataset):  # stands in for a dataset of decoded uint8 HWC images
    def __len__(self):
        return 256

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        h, w = torch.randint(256, 513, (2,), generator=g).tolist()
        return torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=g)


def collate(images):  # pad to the largest image and record each image's ROI
    h_max, w_max = max(im.shape[0] for im in images), max(im.shape[1] for im in images)
    batch = torch.zeros((len(images), h_max, w_max, 3), dtype=torch.uint8)
    for i, im in enumerate(images):
        batch[i, : im.shape[0], : im.shape[1]] = im
    roi = torch.tensor([[0, 0, im.shape[1], im.shape[0]] for im in images], dtype=torch.int32)
    return batch, roi


loader = DataLoader(RandomImages(), batch_size=64, collate_fn=collate, num_workers=2)
for batch, roi in loader:
    batch, roi = batch.cuda(), roi.cuda()
    mirror = torch.randint(0, 2, (batch.shape[0],), dtype=torch.uint8, device="cuda")
    x = hipaugment.fused_rcmn(batch, roi, (224, 224), mirror)  # float32 [N, 3, 224, 224]
    print(tuple(x.shape), x.dtype, f"mean {x.mean().item():+.3f}")
