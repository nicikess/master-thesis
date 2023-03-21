import pandas as pd
from torchmetrics.classification import BinaryAccuracy
import torch
import numpy as np

if __name__ == "__main__":

    # Metrics
    target = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    preds = torch.tensor([[0.1, 0.9, 0.9, 0.1], [0.2, 0.8, 0.8, 0.2]])
    print(np.shape(preds))
    mca = BinaryAccuracy()
    print(mca(preds, target))

    target_transpose = torch.transpose(target, 0, 1)
    print(target)
    print("")
    print(target_transpose)
    preds_transpose = torch.transpose(preds, 0, 1)
    print(preds_transpose)
    mca = BinaryAccuracy(multidim_average='samplewise')
    print(mca(preds_transpose, target_transpose))


    # Evelation model
    # data = pd.read_csv('/ds2/remote_sensing/ben-ge/ben-ge-s'+'/ben-ge-s_climatezones.csv')
    # print(data.head())
    # Climate zone
    # LU/LC


