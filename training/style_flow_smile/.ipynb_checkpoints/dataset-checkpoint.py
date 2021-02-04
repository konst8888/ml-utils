from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import os
import json

class DatasetGenFace(Dataset):
    
    def __init__(self, data_path, mask, transform=None):
        
        self.source_paths = [f for f in os.listdir(data_path) if 'source_' in f]
        self.target_paths = [f for f in os.listdir(data_path) if 'target2_' in f] * 2
        self.source_paths.sort(key=lambda x: int(x.split('_')[3]))
        self.target_paths.sort(key=lambda x: int(x.split('_')[2]))
        self.source_paths = [f for f, m in zip(self.source_paths, mask) if m == 1]
        self.target_paths = [f for f, m in zip(self.target_paths, mask) if m == 1]
        self.source_paths = list(map(lambda x: os.path.join(data_path, x), self.source_paths))
        self.target_paths = list(map(lambda x: os.path.join(data_path, x), self.target_paths))
        self.transform = transform
        self.probas = {
            'HorizontalFlip': 0.5,
            'ShiftScaleRotate': 0.3,
            'RandomBrightness': 0.1,
            'Blur': 0.05,
            'JpegCompression': 0.05,
            'GaussNoise': 0.05,
            'ToGray': 0.05
        }
        if False:
            self.probas = {k: 0. for k in self.probas}
        self.seed = 0
        
    def filter_index(self, ixs, is_seed):
        if not is_seed:
            self.source_paths = np.array(self.source_paths)[ixs]
            self.target_paths = np.array(self.target_paths)[ixs]
        else:
            self.source_paths = np.array([f for f in self.source_paths if int(f.split('_')[-7]) in ixs])
            self.target_paths = np.array([f for f in self.target_paths if int(f.split('_')[-7]) in ixs])
            
        print('Len after filtering: ', len(self.source_paths))
        
        ixs_eq = [ix for ix, (s, t) in enumerate(zip(self.source_paths, self.target_paths)) 
                   if s.split('_')[-1].replace('.jpg', '') == t.split('_')[-1].replace('.jpg', '')]
        ixs_not_eq = list(set(range(len(self.source_paths))) - set(ixs_eq))
        ixs_final = ixs_not_eq + ixs_eq[::2]
        self.source_paths = self.source_paths[ixs_final]
        self.target_paths = self.target_paths[ixs_final]
        print('Len after drop equals: ', len(self.source_paths))

        self.source_paths = self.source_paths.tolist()
        self.target_paths = self.target_paths.tolist()
        
        with open('ixs_final.txt', 'w') as f:
            for ix in ixs_final:
                f.write(str(ix) + '\n')
        
    def __len__(self):
        return len(self.source_paths)
        
    def __getitem__(self, idx):
        source_path = self.source_paths[idx]
        target_path = self.target_paths[idx]
        
        source_pic = Image.open(source_path)
        target_pic = Image.open(target_path)
        
        pp = {k: 1.0 if random.random() < v else 0.0 for k, v in self.probas.items()}
        source_tensor = self.transform(source_pic, pp=pp, seed=self.seed)
        target_tensor = self.transform(target_pic, pp=pp, seed=self.seed)
        self.seed += 1
        seed = int(target_path.split('_')[-7])
        return seed, source_tensor, target_tensor