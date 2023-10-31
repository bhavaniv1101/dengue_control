import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

from complete_program import SegDataset, FocalLoss, mIoULoss, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acc(label, predicted):
    """
    Calculates the segmentation accuracy.
    """
    seg_acc = (
        label.cpu() == torch.argmax(predicted, axis=1).cpu()
    ).sum() / torch.numel(label.cpu())
    return seg_acc


def train_class(cls: str):
    print("Class: " + cls)

    # transformations
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])

    if cls == "Road":
        images = "/for_road/images"
    elif cls == "Building":
        images = "/for_building/images"
    else:
        # For water
        images = "/kaggle_dataset/Images"
        # images = "/no_road_building/images"

    dataset = SegDataset(
        training=True,
        transform=t,
        img_names="C:/Users/sivar/Downloads/SIP-Classwise-Segmentation/SIP-Classwise-Segmentation"
        + images,
        mask_names=f"C:/Users/sivar/Downloads/SIP-Classwise-Segmentation/SIP-Classwise-Segmentation"
                   f"/kaggle_dataset/Masks",
        # mask_names=f"C:/Users/sivar/Downloads/SIP-Classwise-Segmentation/SIP-Classwise-Segmentation"
        #            f"/class_masks/{cls}",
    )

    # dataset = ClasswiseSegDataset(
    #     root='C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation', split_idx=150
    # )

    print(f"Length of dataset: {len(dataset)}")

    test_num = int(0.1 * len(dataset))
    print(f"Test data : {test_num}")
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_num, test_num],
        generator=torch.Generator().manual_seed(101),
    )

    # train_dataset, test_dataset = dataset, dataset

    BATCH_SIZE = 4
    print(f"Batch size: {BATCH_SIZE}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    criterion = FocalLoss(gamma=3 / 4).to(device)
    loss_function = "Focal loss"

    # criterion = mIoULoss(n_classes=28).to(device)
    # loss_function = 'IoU loss'

    print(f"Loss function: {loss_function}")

    min_loss = torch.tensor(float("inf"))

    model = UNet(n_channels=3, n_classes=28, bilinear=True).to(device)
    # Load saved model and continue training from there
    model.load_state_dict(torch.load("saved_models/Water.pt", map_location=torch.device("cpu")))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # os.chdir('C:/Users/bhava/Documents/SIP AAI-06')
    os.makedirs("./saved_models", exist_ok=True)

    N_EPOCHS = 100
    N_DATA = len(train_dataset)
    N_TEST = len(test_dataset)

    plot_losses = []
    scheduler_counter = 0

    for epoch in range(N_EPOCHS):
        # training
        model.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):
            pred_mask = model(x.to(device))
            loss = criterion(pred_mask, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y, pred_mask).numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch + 1,
                    N_EPOCHS,
                    batch_i + 1,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )

        scheduler_counter += 1

        # testing
        model.eval()
        val_loss_list = []
        val_acc_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():
                pred_mask = model(x.to(device))
            val_loss = criterion(pred_mask, y.to(device))
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(acc(y, pred_mask).numpy())

        print(
            " epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}".format(
                epoch + 1,
                np.mean(loss_list),
                np.mean(acc_list),
                np.mean(val_loss_list),
                np.mean(val_acc_list),
            )
        )

        plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        if is_best:
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(
                model.state_dict(),
                "./saved_models/{}_unet_epoch_{}_{:.5f}.pt".format(
                    cls, epoch, np.mean(val_loss_list)
                ),
            )

        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"Lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0

    # plot loss
    plt.figure(figsize=(15, 15))
    plot_losses = np.array(plot_losses)
    plt.plot(plot_losses[:, 0], plot_losses[:, 1], color="b", linewidth=4)
    plt.plot(plot_losses[:, 0], plot_losses[:, 2], color="r", linewidth=4)
    plt.title(loss_function, fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.grid()
    plt.legend(["training", "validation"])
    plt.savefig(f"./{cls}_loss_graph.png")

    model.eval()

    for batch_i, (x, y) in enumerate(test_dataloader):
        for j in range(len(x)):
            result = model(x.to(device)[j : j + 1])
            mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
            im = np.moveaxis(x.to(device)[j].cpu().detach().numpy(), 0, -1).copy() * 255
            im = im.astype(int)
            gt_mask = y[j]

            plt.figure(figsize=(12, 12))

            plt.subplot(1, 3, 1)
            im = np.moveaxis(x.to(device)[j].cpu().detach().numpy(), 0, -1).copy() * 255
            im = im.astype(int)
            plt.title("Image")
            plt.imshow(im)

            plt.subplot(1, 3, 2)
            plt.title("Ground truth mask")
            plt.imshow(gt_mask)

            plt.subplot(1, 3, 3)
            plt.title("Mask")
            plt.imshow(mask)
            plt.savefig(f"{cls}_output_{batch_i}_{j}")


if __name__ == "__main__":
    classes = [
        "Water",
        # "Building",
        # "Land",
        # "Road",
        # "Vegetation",
        # "Tree",
        # "Grass",
        # "Rocks",
        # "Window",
        # "Door",
        # "Fence",
        # "Person",
        # "Car",
        # "Bicycle",
    ]

    for cls_name in classes:
        train_class(cls_name)
