from utils import AUROC_Score
import numpy as np


def process(file_data):
    a = np.load(file_data)[:3000]

    score = np.empty(len(a))

    for i in range(len(a)):
        score[i] = np.mean(a[i] == a[i][0])

    return score


if __name__ == "__main__":
    AUROC_Score(
        process("saved_np/WaNet/tiny_bd.npy"),
        process("saved_np/WaNet/tiny_benign.npy"),
        "tiny_imagenet",
    )
