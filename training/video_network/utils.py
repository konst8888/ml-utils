from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import random
import torch
import os

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