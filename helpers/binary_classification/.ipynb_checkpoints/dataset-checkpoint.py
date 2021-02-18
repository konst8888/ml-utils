from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import itertools
import tqdm

class BinaryClassDataset:
        
    def __init__(self, data, target_name='label', class_map=None):
        self.X = data.drop(columns={target_name})
        self.y = data[target_name]
        if class_map is not None:
            pos, neg = class_map
            self.y = self.y.map({pos: 1, neg: 0})
        self.processed_columns = []
        
    def __hstack(self, A):
        self.X = pd.concat([self.X, A], axis=1)
    
    def __drop_processed_columns(self):
        if len(self.processed_columns) > 0:
            self.X.drop(columns=self.processed_columns, inplace=True)
            self.processed_columns = []
        
    def __add_comb(self, comb):
        if len(comb.split('-')) == 1:
            #comb = '-' + comb
            #self.X[comb] = ''
            #self.X[comb] = self.X[comb[1:]]
            return
        self.X[comb] = ''
        for col in comb.split('-'):
            self.X[comb] = self.X[comb] + self.X[col].apply(lambda x: '-' + str(x))
        
    def __slice_data(self, A, idxs):
        if isinstance(A, pd.DataFrame):
            cols = A.columns
            A = A.to_numpy()
            A = A[idxs]
            A = pd.DataFrame(A, columns=cols)
        elif isinstance(A, pd.Series):
            A = A.to_numpy()
            A = A[idxs]
            A = pd.Series(A)
    	
        return A
        
    def __dropna(self):
        self.X['label'] = self.y.to_numpy()
        self.X.dropna(axis=1, inplace=True)
        self.y = self.X.label
        del self.X['label']
        
    def get_data(self, test_size):
        self.__drop_processed_columns()
        self.__dropna()
        X_train, y_train, X_test, y_test = self.stratified_split(test_size=test_size)
        return X_train, y_train, X_test, y_test
    
    def train_test_split(self, final=True, **params):
        """
        params:
        test_size, random_state, shuffle
        """
        if final:
            self.__drop_processed_columns()
        X_train, y_train, X_test, y_test = train_test_split(self.X, self.y, **params)
        
        return (X_train, y_train, X_test, y_test)
    
    def stratified_split(self, **params):
        """
        params:
        test_size, random_state, shuffle
        """
        test_size = params['test_size']
        n_splits = int(1. / test_size)
        skf = StratifiedKFold(n_splits=n_splits)
        train_index, test_index = next(skf.split(self.X, self.y))
        X_train, X_test = self.__slice_data(self.X, train_index), self.__slice_data(self.X, test_index)
        y_train, y_test = self.__slice_data(self.y, train_index), self.__slice_data(self.y, test_index)    
        
        return (X_train, y_train, X_test, y_test)
    
    def one_hot_encoding(self, columns, cross_prod_dim=1):
        
        for col in columns:
            self.X[col] = self.X[col].astype(str)

        combs = list(itertools.combinations(columns, cross_prod_dim))
        combs = ['-'.join(comb) for comb in combs]

        for comb in tqdm.tqdm(combs, total=len(combs)):
            self.__add_comb(comb)
            #print(self.X[comb])
            one_hot = pd.get_dummies(self.X[comb])
            one_hot.columns = [comb + '_' + str(c) for c in one_hot.columns]
            self.__hstack(one_hot)
        
        #for col in combs:
        #    one_hot = pd.get_dummies(self.X[col])
        #    one_hot.columns = [col + '_' + str(c) for c in one_hot.columns]
        #    self.__hstack(self.X, one_hot)
        if cross_prod_dim == 1 and len(columns) == 1:
            drop_columns = columns
        else:
            drop_columns = combs + columns
        self.processed_columns.extend(drop_columns)
        

    def cat_label_transform(self, transform_size, columns, aggs=['mean'], **params):
        """
        aggs: [mean, std, median]
        params:
        random_state, shuffle
        """
        
        params['test_size'] = transform_size
        X_transform, y_transform, X_train, y_train = self.stratified_split(**params)
        
        for col in tqdm.tqdm(columns, total=len(columns)):
            data_slice = pd.DataFrame()
            data_slice[col] = X_train[col].astype(str).to_numpy()
            data_slice['label'] = y_train.to_numpy()
            uniques = data_slice[col].unique()
            mapping = dict()
            for agg in aggs:
                mapping[agg] = {val: np.nan for val in uniques}
            for val in uniques:
                if 'mean' in aggs:
                    mapping['mean'][val] = float(data_slice[data_slice[col] == val].label.mean())
                if 'std' in aggs:
                    mapping['mean'][val] = float(data_slice[data_slice[col] == val].label.std())
                if 'median' in aggs:
                    mapping['mean'][val] = float(data_slice[data_slice[col] == val].label.median())
            for agg in aggs:
                X_transform[col + f'_{agg}_y'] = X_transform[col].map(mapping[agg])
            
        self.X = X_transform
        self.y = y_transform
            
        self.processed_columns.extend(columns)
        
    def normalize(self, columns):
    	for col in columns:
    	    mx = self.X[col].max()
    	    mn = self.X[col].min()
    	    self.X[col] = self.X[col].apply(lambda x: (x - mn) / (mx - mn))