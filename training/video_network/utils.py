from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

def get_split(data_path, n_classes=2, test_size=0.5):

    labels = []
    for n in range(n_classes):
        labels += [n] * len(next(os.walk(os.path.join(data_path, str(n))))[1])

    labels = np.array(labels)
    #print(next(os.walk(os.path.join(data_path, str(0)))))
    skf = StratifiedKFold(n_splits=int(1 / test_size))
    train_index, test_index = next(skf.split(labels, labels))

    return train_index, test_index