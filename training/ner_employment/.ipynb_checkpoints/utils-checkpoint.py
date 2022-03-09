from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch
from typing import List, Any, Tuple
import random
import torch
import os

def collate_fn(examples: List[Any]) -> Tuple[torch.Tensor, ...]:
    """Batching examples.

    Parameters
    ----------
    examples : List[Any]
        List of examples

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Tuple of hash tensor, length tensor, and label tensor
    """

    projection = []
    masks = []
    labels = []
    tokens = []
    idxs = []
    
    for example in examples:
        if not isinstance(example, tuple):
            projection.append(np.asarray(example))
        else:
            projection.append(np.asarray(example[0]))
            masks.append(example[1])
            labels.append(example[2])
            tokens.append(example[3])
            idxs.append(example[4])
    #lengths = torch.from_numpy(np.asarray(list(map(len, examples)))).long()
    masks = torch.LongTensor(masks)
    projection_tensor = np.zeros(
        (len(projection), max(map(len, projection)), len(projection[0][0]))
    )
    for i, doc in enumerate(projection):
        projection_tensor[i, : len(doc), :] = doc
    return (
        torch.from_numpy(projection_tensor).float(),
        masks,
        torch.from_numpy(np.asarray(labels)),
        np.array(tokens),
        idxs,
    )

def load_state_dict(model, model_path, device, source=''):
    
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    if source == 'jester':
        pretrained_renamed_dict = {k.replace('module.', ''): v for k, v in pretrained_dict['state_dict'].items()}
        pretrained_dict = pretrained_renamed_dict

    model_dict_new = model_dict.copy()
    counter = 0
    for k, v in pretrained_dict.items():
        if k in model_dict_new and np.all(v.size() == model_dict_new[k].size()):
            model_dict_new.update({k: v})
            counter += 1
    print(f'Loaded: {counter}/{len(model_dict)}')
    model.load_state_dict(model_dict_new)
    return model

def get_split(data_path, csv_path, n_classes=2, test_size=0.1):

    random.seed(0)
    np.random.seed(0)
    
    labels = []
    #for n in range(n_classes):
        #labels += [n] * len(os.listdir(os.path.join(data_path, str(n))))
        #labels += [n] * len(next(os.walk(os.path.join(data_path, str(n))))[1])

    #csv_path = 'non_personal_video_train.csv'
    data = pd.read_csv(csv_path)

    labels = data.label.to_numpy()
    #labels = [0, 1] * 8040
    #labels = np.array(labels)
    #print(next(os.walk(os.path.join(data_path, str(0)))))
    skf = StratifiedKFold(n_splits=int(1 / test_size))
    train_index, test_index = next(skf.split(labels, labels))

    return train_index, test_index