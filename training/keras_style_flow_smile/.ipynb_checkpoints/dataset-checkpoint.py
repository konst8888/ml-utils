import tensorflow as tf
import numpy as np
import os

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_path, mask, batch_size, loader, size):
        self.source_paths = [f for f in os.listdir(data_path) if 'source_' in f]
        self.target_paths = [f for f in os.listdir(data_path) if 'target2_' in f] * 2
        self.source_paths.sort(key=lambda x: int(x.split('_')[-7]))
        self.target_paths.sort(key=lambda x: int(x.split('_')[-7]))
        self.source_paths = [f for f, m in zip(self.source_paths, mask) if m == 1]
        self.target_paths = [f for f, m in zip(self.target_paths, mask) if m == 1]
        self.source_paths = list(map(lambda x: os.path.join(data_path, x), self.source_paths))
        self.target_paths = list(map(lambda x: os.path.join(data_path, x), self.target_paths))
        self.batch_size = batch_size
        self.loader = loader
        self.y = size[0]
        self.x = size[1]

    def __len__(self):
        return len(self.target_paths) // self.batch_size
    
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
        
    def __getitem__(self, index):
        X, y = self.__get_data(index)
        return X, y

    def on_epoch_end(self):
        pass

    def __get_data1(self, idx):
        print('here')
        source_path = self.source_paths[idx]
        target_path = self.target_paths[idx]
        seed = int(target_path.split('_')[-7])

        source_tensor, target_tensor = self.loader(source_path, target_path)

        return source_tensor, target_tensor

    def __get_data(self, idx):
        source_path = self.source_paths[idx]
        target_path = self.target_paths[idx]

        X = np.empty((self.batch_size, self.x, self.y, 3))
        y = np.empty((self.batch_size, self.x, self.y, 3))

        for i, id in enumerate(range(idx * self.batch_size, (idx+1) * self.batch_size)):
            source_tensor, target_tensor = self.loader(source_path, target_path)
            X[i, ] = source_tensor
            y[i, ] = target_tensor

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        return X, y
