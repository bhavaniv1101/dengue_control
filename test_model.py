from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

from complete_program import SegDataset, FocalLoss, mIoULoss, UNet


def run():
    """Run code."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=3, n_classes=28, bilinear=True).to(device)
    model.load_state_dict(torch.load("saved_models/Water.pt", map_location=torch.device("cpu")))
    model.eval()

    test_dataset = SegDataset(
        training=True,
        transform=False,
        img_names="C:/Users/sivar/Downloads/SIP-Classwise-Segmentation/SIP-Classwise-Segmentation"
                  + "/test_dataset",
        mask_names=None,
    )
    print(f"Length of dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=1
    )

    for batch_i, (x, y) in enumerate(test_dataloader):
        print(f"Batch {batch_i}")
        for j in range(len(x)):
            result = model(x.to(device)[j: j + 1])
            mask = torch.argmax(result, dim=1).cpu().detach().numpy()[0]
            # im = np.moveaxis(x.to(device)[j].cpu().detach().numpy(), 0, -1).copy() * 255
            # im = im.astype(int)
            # gt_mask = y[j]

            plt.figure(figsize=(12, 12))

            plt.subplot(1, 3, 1)
            im = np.moveaxis(x.to(device)[j].cpu().detach().numpy(), 0, -1).copy() * 255
            im = im.astype(int)
            plt.title("Image")
            plt.imshow(im)

            # plt.subplot(1, 3, 2)
            # plt.title("Ground truth mask")
            # plt.imshow(gt_mask)

            plt.subplot(1, 3, 3)
            plt.title("Mask")
            plt.imshow(mask)
            plt.savefig(f"output_{batch_i}_{j}")


if __name__ == '__main__':
    run()
