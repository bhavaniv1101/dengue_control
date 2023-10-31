import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(fname="kaggle_dataset_output.txt"):
    """Plots validation accuracy versus epoch."""
    val_acc = []
    with open(fname, "r") as file:
        for line in file:
            if line.startswith("[Epoch"):
                val_acc.append(float(line[-4 :]))
    val_acc = np.array(val_acc)
    fig, axs = plt.subplots()
    axs.plot(1 + np.arange(val_acc.size), val_acc)
    axs.set_xlabel("Training epoch")
    axs.set_ylabel("Accuracy on validation set")
    axs.set_title("Segmentation accuracy for water")
    axs.grid(True)
    fig.savefig("val_acc_vs_epoch.png", dpi=200)


if __name__ == "__main__":
    plot_accuracy()
