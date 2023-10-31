import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from glob import glob
import sys
import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from scipy import ndimage
from torch.autograd import Variable

# 04-07-2022 22:32 original version

# adjust learning rate - increase
# decrease data size - 100
# train for more iterations
# train and test with same dataset - acc should be very high - to make sure model works
# split aerial pic into small patches - 500 patches
# delete ones with blank space

# TASK:seperate model for each class - class object white, rest black
# IDEA: check each pixel of the MASK to see if it is matching the object color,
# if it is, assign a specific color to it; if it is, not make it black
#
# TASK: at the end, run every model on a new image and sum up the results
# IDEA: creat a mask of zeros of the size of the image, if a pixel is black, leave it as zero,
# if it is not black, add it to the mask and repeat for each image from the different models.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patchify: bool = False,
        patch_size: int = None,
        step: int = None,
        root: str = "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/100_dataset",
        training: bool = True,
        transform: bool = None,
        img_names: str = None,
        mask_names: str = None,
    ) -> None:
        """
        Initializes the dataset and and sets BGR colors for each class.

        :param patchify: if True, images will be split into patches; if False, images are left as they are
        :param patch_size: size of the patch (if patchify is True)
        :param step: step size to determine the starting point of the next patch (if patchify is True)
        :param root: path for the original dataset
        :param training: True if training should be done, False otherwise
        :param transform: transformations to be applied to the images
        """
        super(SegDataset, self).__init__()
        self.root = root
        self.training = training
        self.transform = transform
        if img_names is None:
            self.IMG_NAMES = sorted(glob(self.root + "/images/*"))
        else:
            self.IMG_NAMES = glob(img_names + "/*")
        if mask_names is None:
            self.MASK_NAMES = sorted(glob(self.root + "/masks/*"))
        else:
            self.MASK_NAMES = glob(mask_names + "/*")
        # classes and colors for the masks
        self.BGR_classes = {
            "Water": [255, 0, 0],  # blue
            "Pool": [89, 50, 0],  # dark blue
            "Land": [246, 41, 132],  # purple
            "Dirt": [0, 76, 130],  # brown
            "Gravel": [87, 103, 112],  # grey-brown
            "Rocks": [30, 41, 48],  # dark blackish-brown
            "Road": [228, 193, 110],  # light blue
            "Paved area": [128, 64, 128],  # magenta purple
            "Grass": [0, 102, 0],  # green
            "Vegetation": [41, 169, 226],  # yellow-orange-brown
            "Tree": [0, 51, 51],  # dark olive-green
            "Bald-tree": [190, 250, 190],  # light green
            "Building": [152, 16, 60],  # dark violet
            "Roof": [70, 70, 70],  # dark grey
            "Wall": [156, 102, 102],  # bluish-purple
            "Window": [12, 228, 254],  # bright yellow
            "Door": [12, 148, 254],  # orange
            "Fence": [153, 153, 190],  # dull pink
            "Fence-pole": [153, 153, 153],  # light grey
            "Car": [150, 143, 9],  # dull teal blue
            "Bicycle": [32, 11, 119],  # dark red / maroon
            "Person": [96, 22, 255],  # hot pink
            "Dog": [0, 51, 102],  # mud brown
            "AR-marker": [146, 150, 112],  # dull greyish-blue
            "Background": [0, 0, 0],  # black
            "Obstacle": [115, 135, 2],  # dark bluish-green
            "Conflicting": [0, 0, 255],  # red
            "Unlabeled": [255, 255, 255],
        }  # white

        # list of classes
        self.bin_classes = [
            "Water",
            "Pool",
            "Land",
            "Dirt",
            "Gravel",
            "Rocks",
            "Road",
            "Paved area",
            "Grass",
            "Vegetation",
            "Tree",
            "Bald-tree",
            "Building",
            "Roof",
            "Wall",
            "Window",
            "Door",
            "Fence",
            "Fence-pole",
            "Car",
            "Bicycle",
            "Person",
            "Dog",
            "AR-marker",
            "Background",
            "Obstacle",
            "Conflicting",
            "Unlabeled",
        ]

        def is_single_color(image, color: list) -> bool:
            """
            Checks if all the pixels in image are of the same given color.

            :param image: the image which is to be checked
            :param color: list of length three with the BGR values for the color

            :return: True if all pixels are of the same color and False otherwise
            """
            pixels = np.all(
                image == color, axis=-1
            )  # array of bool values - True if pixel == color, False otherwise
            for row in pixels:
                for color_pix in row:
                    if not color_pix:
                        return False
            return True

        def patch_image(
            img_path: str,
            img_num: int,
            x_size: int,
            y_size: int,
            step: int,
            check_colors: bool = False,
            color_list: list = None,
        ) -> list:
            """
            Reads image from the given path and splits it into multiple patches of size x_size by y_size,
            with the given step size. The patches are saved in a new folder.

            If check_colors is True, it checks each patch to see if all the pixels are of the same given color;
            this is done for every color in color_list. If a patch is of a single color, it is not saved.

            :param img_path: path of the image that needs to be split into patches
            :param img_num: image number (for naming the patches while saving)
            :param x_size: x dimension for the patch size (length)
            :param y_size: y dimension for the patch size (height)
            :param step: step size to determine the starting point of the next patch
            :param check_colors: True if patches need to be checked to see if all pixels same color; False otherwise
            :param color_list: list of colors to check (if check_colors is True)

            :return: list of the names of images that were NOT saved
            """
            image = cv2.imread(img_path)
            image = cv2.resize(image, (512, 512))  # HWC

            patch_num = 0
            remove_list = []  # list of names of images that have not been saved

            for start_x in range(0, image.shape[1], step):
                for start_y in range(0, image.shape[0], step):
                    # create a patch by slicing the array
                    patch = image[
                        start_y : start_y + y_size, start_x : start_x + x_size
                    ]
                    # if the patch is bigger than half the specified size, resize it to 512 x 512
                    if patch.shape[0] > y_size / 2 and patch.shape[1] > x_size / 2:
                        patch = cv2.resize(patch, (512, 512))

                        if not check_colors:
                            cv2.imwrite(f"patch_img_{img_num}_{patch_num}.jpg", patch)

                        # if the patch needs to be checked to see if all pixels are the same color,
                        # go through the list of colors and check the patch for each color
                        if check_colors:
                            keep = True
                            for color in color_list:
                                if is_single_color(patch, color):
                                    # if a patch is of single color, add it to list of images that have not been saved
                                    remove_list.append(
                                        f"patch_img_{img_num}_{patch_num}.jpg"
                                    )
                                    keep = False
                                    break
                            # if the patch is not of a single color, save it
                            if keep:
                                cv2.imwrite(
                                    f"patch_img_{img_num}_{patch_num}.jpg", patch
                                )

                        patch_num += 1

            return remove_list

        def save_patches() -> None:
            """
            Patchifies all aerial/satellite images and masks in the dataset and saves them in a new folder - 'patched'.
            Images (and masks) whose names start with 0 are not patchified as they are not aerial images.
            """
            remove_list = []  # list of images/masks that have not been saved

            # patchify images
            for img_num, img_path in enumerate(self.IMG_NAMES):
                img_path = img_path.replace("\\", "/")
                # go to the directory where patchififed images should be saved
                os.chdir(
                    "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/patched/images"
                )
                if not (
                    img_path.startswith(
                        "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/100_dataset/images/0"
                    )
                ):
                    remove_list += patch_image(
                        img_path,
                        img_num,
                        patch_size,
                        patch_size,
                        step,
                        check_colors=True,
                        color_list=[[255, 255, 255]],
                    )
                # if the image does not need to be patchified, it is resized to 512 x 512 (to match the patches)
                # and it is saved in the same directory
                else:
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (512, 512))  # HWC
                    cv2.imwrite(f"patch_img_{img_num}_0.jpg", image)

            # patchify masks - same process as images
            for mask_num, mask_path in enumerate(self.MASK_NAMES):
                mask_path = mask_path.replace("\\", "/")
                os.chdir(
                    "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/patched/masks"
                )
                if not (
                    mask_path.startswith(
                        "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/100_dataset/masks/0"
                    )
                ):
                    remove_list += patch_image(
                        mask_path,
                        mask_num,
                        patch_size,
                        patch_size,
                        step,
                        check_colors=True,
                        color_list=[
                            [0, 0, 0],
                            [41, 169, 226],
                            [246, 41, 132],
                            [58, 221, 254],
                        ],
                    )
                else:
                    mask = cv2.imread(mask_path)
                    mask = cv2.resize(mask, (512, 512))  # HWC
                    cv2.imwrite(f"patch_img_{mask_num}_0.jpg", mask)

            remove_list.append(
                "patch_img_99_24.jpg"
            )  # to make data size 1500 instead of 1501

            print("Images removed: ", len(remove_list))
            # ensure that the images and masks corresponding to those that were not saved are removed from the directory
            for name in remove_list:
                img_path = (
                    "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/patched/images/"
                    + name
                )
                if os.path.exists(img_path):
                    os.remove(img_path)
                mask_path = (
                    "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/patched/masks/"
                    + name
                )
                if os.path.exists(mask_path):
                    os.remove(mask_path)

        # if the images should be patchified, make the 'patched' folder the data source
        if patchify:
            save_patches()
            self.IMG_NAMES = sorted(
                glob(
                    "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/patched/images/*"
                )
            )
            self.MASK_NAMES = sorted(
                glob(
                    "C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/patched/masks/*"
                )
            )

    def __getitem__(self, idx: int):
        """
        Returns the image and mask at a specified index, with some transformations applied to it.

        :param idx: index of the image and mask that need to be retrieved
        :return: the image and mask as tensors
        """
        img_path = self.IMG_NAMES[idx]
        image = cv2.imread(img_path)

        # HACK!
        if idx < len(self.MASK_NAMES):
            mask_path = self.MASK_NAMES[idx]
            mask = cv2.imread(mask_path)
        else:
            mask = np.zeros_like(image)

        # give an index “label” to specific BGR values to make it easier for the model to predict using calculations
        cls_mask = np.zeros(mask.shape)
        cls_mask[mask == self.BGR_classes["Water"]] = self.bin_classes.index("Water")
        cls_mask[mask == self.BGR_classes["Pool"]] = self.bin_classes.index("Pool")
        cls_mask[mask == self.BGR_classes["Land"]] = self.bin_classes.index("Land")
        cls_mask[mask == self.BGR_classes["Dirt"]] = self.bin_classes.index("Dirt")
        cls_mask[mask == self.BGR_classes["Gravel"]] = self.bin_classes.index("Gravel")
        cls_mask[mask == self.BGR_classes["Rocks"]] = self.bin_classes.index("Rocks")
        cls_mask[mask == self.BGR_classes["Road"]] = self.bin_classes.index("Road")
        cls_mask[mask == self.BGR_classes["Paved area"]] = self.bin_classes.index(
            "Paved area"
        )
        cls_mask[mask == self.BGR_classes["Grass"]] = self.bin_classes.index("Grass")
        cls_mask[mask == self.BGR_classes["Vegetation"]] = self.bin_classes.index(
            "Vegetation"
        )
        cls_mask[mask == self.BGR_classes["Tree"]] = self.bin_classes.index("Tree")
        cls_mask[mask == self.BGR_classes["Bald-tree"]] = self.bin_classes.index(
            "Bald-tree"
        )
        cls_mask[mask == self.BGR_classes["Building"]] = self.bin_classes.index(
            "Building"
        )
        cls_mask[mask == self.BGR_classes["Roof"]] = self.bin_classes.index("Roof")
        cls_mask[mask == self.BGR_classes["Wall"]] = self.bin_classes.index("Wall")
        cls_mask[mask == self.BGR_classes["Window"]] = self.bin_classes.index("Window")
        cls_mask[mask == self.BGR_classes["Door"]] = self.bin_classes.index("Door")
        cls_mask[mask == self.BGR_classes["Fence"]] = self.bin_classes.index("Fence")
        cls_mask[mask == self.BGR_classes["Fence-pole"]] = self.bin_classes.index(
            "Fence-pole"
        )
        cls_mask[mask == self.BGR_classes["Car"]] = self.bin_classes.index("Car")
        cls_mask[mask == self.BGR_classes["Bicycle"]] = self.bin_classes.index(
            "Bicycle"
        )
        cls_mask[mask == self.BGR_classes["Person"]] = self.bin_classes.index("Person")
        cls_mask[mask == self.BGR_classes["Dog"]] = self.bin_classes.index("Dog")
        cls_mask[mask == self.BGR_classes["AR-marker"]] = self.bin_classes.index(
            "AR-marker"
        )
        cls_mask[mask == self.BGR_classes["Background"]] = self.bin_classes.index(
            "Background"
        )
        cls_mask[mask == self.BGR_classes["Obstacle"]] = self.bin_classes.index(
            "Obstacle"
        )
        cls_mask[mask == self.BGR_classes["Conflicting"]] = self.bin_classes.index(
            "Conflicting"
        )
        cls_mask[mask == self.BGR_classes["Unlabeled"]] = self.bin_classes.index(
            "Unlabeled"
        )
        cls_mask = cls_mask[:, :, 0]

        # apply transformations
        if self.training:
            if self.transform:
                image = transforms.functional.to_pil_image(image)
                image = self.transform(image)
                image = np.array(image)

            # 90 degree rotation
            if np.random.rand() < 0.5:
                angle = np.random.randint(4) * 90
                image = ndimage.rotate(image, angle, reshape=True)
                cls_mask = ndimage.rotate(cls_mask, angle, reshape=True)

            # vertical flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 0)
                cls_mask = np.flip(cls_mask, 0)

            # horizonal flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 1)
                cls_mask = np.flip(cls_mask, 1)

        image = cv2.resize(image, (512, 512)) / 255.0
        cls_mask = cv2.resize(cls_mask, (512, 512))
        image = np.moveaxis(image, -1, 0)

        return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.IMG_NAMES)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Focal loss adds an extra term to reduce the impact of correct predictions and focus on incorrect examples.
# Gamma is a hyperparameter that specifies how powerful this reduction will be.
# Alpha is a weight hyperparameter for different classes and is a way to balance the loss for unbalanced classes.


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        # if alpha is either float or int, make self.alpha a Tensor object using a list [alpha, 1 - alpha]
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        # if alpha is a list, make self.alpha a Tensor object using the list
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N -> batch dimension (number of images present),
            # C -> channel dimension (number of channels present),
            # H -> image height, W -> image width
            # the size -1 is inferred from other dimensions
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)  # gather values along an axis specified by dim
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())  # tensor with exponential of elements in logpt

        if self.alpha is not None:
            # make alpha the same data type as input
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt  # calculate focal loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# The numerator is the overlap between the predicted and ground-truth masks, and the denominator is the union of them.
# The IoU is calculated by dividing these two numbers, with values closer to one indicating more accurate predictions.
# The loss is calculated by subtracting the IoU from 1.


class mIoULoss(nn.Module):
    def __init__(self, n_classes=28):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        """
        Return a tensor that has zeros everywhere except where the index of last dimension
        matches the corresponding value of the input tensor, in which case it will be 1.
        """
        n, h, w = tensor.size()
        one_hot = (
            torch.zeros(n, self.classes, h, w)
            .to(tensor.device)
            .scatter_(1, tensor.view(n, 1, h, w), 1)
        )
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        # sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        # sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        # return average loss over classes and batch
        return 1 - loss.mean()


def acc(label, predicted):
    """
    Calculates the segmentation accuracy.

    :param label: the correct mask for the image
    :param predicted: the mask produced by the model
    :return: the accuracy of the prediction
    """
    seg_acc = (
        label.cpu() == torch.argmax(predicted, axis=1).cpu()
    ).sum() / torch.numel(label.cpu())
    return seg_acc


if __name__ == "__main__":
    # transformations
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])
    # dataset = SegDataset(patchify=True, patch_size=100, step=100,
    #                      root='C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/100_dataset',
    #                      training=True, transform=t)
    dataset = SegDataset(
        patchify=False,
        root="C:/Users/bhava/Documents/SIP AAI-06/UNet_Aerial_Segmentation/data_without_road_&_building",
        training=True,
        transform=t,
    )

    print(f"Length of dataset: {len(dataset)}")

    test_num = len(dataset)
    print(f"Test data : {test_num}")
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_num, test_num],
    #                                                             generator=torch.Generator().manual_seed(101))

    train_dataset, test_dataset = dataset, dataset

    BATCH_SIZE = 5
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    os.chdir("C:/Users/bhava/Documents/SIP AAI-06")
    os.makedirs("./saved_models", exist_ok=True)

    N_EPOCHS = 10
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
                "C:/Users/bhava/Documents/SIP AAI-06/saved_models/unet_epoch_{}_{:.5f}.pt".format(
                    epoch, np.mean(val_loss_list)
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
    plt.show()

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
            plt.show()
