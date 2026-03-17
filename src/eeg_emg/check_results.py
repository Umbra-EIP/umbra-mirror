import matplotlib.pyplot as plt
import numpy as np


def view_results(filename="eeg2emg_preds.npz"):
    try:
        data = np.load(filename)
        preds = data["preds"]
        trues = data["trues"]
    except FileNotFoundError:
        print(f"file {filename} doesn't exist. have you finished training ?")
        return

    print(f"loading {len(preds)} test windows.")

    indices = np.random.choice(len(preds), 3, replace=False)

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(indices):
        plt.subplot(3, 1, i + 1)
        plt.plot(trues[idx, 0, :], label="True EMG (Target)", color="black", alpha=0.7)
        plt.plot(preds[idx, 0, :], label="IA predictions", color="red", linestyle="--")
        plt.title(f"Example #{idx} - EMG canal 0")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    view_results()
