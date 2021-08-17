from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RoundData(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        for col in self.columns:
            X[col] = X[col].apply(round)
        return X, y
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
class SMOTENCR(SMOTENC):
    
    def __init__(self, categorical_features, random_state, round_columns):
        super().__init__(categorical_features=categorical_features, random_state=random_state)
        self.round_columns = round_columns
        
    def fit_resample(self, X, y):
        X, y = super().fit_resample(X, y)
        for col in self.round_columns:
            X[col] = X[col].apply(round)
        return X, y

class Classification:
    
    def __init__(self, lib, clf, upsample=True, balance_test=True, seed=123, **params):
        if lib == 'xgb':
            default_params = {
                "objective":"binary:logistic",
                'colsample_bytree': 0.75,
                'learning_rate': 0.1,
                'max_depth': 5, 
                'alpha': 10
            }
            for p in default_params:
                if p not in params:
                    params[p] = default_params[p]
                    
        self.model = clf(**params)
        self.lib = lib
        self.upsample = upsample
        self.balance_test = balance_test
        self.seed = seed
        
    def fit(self, *args):
        """
        X, y - training dataset and labels
        or
        BinaryClassDataset object
        """
        if len(args) == 1:
            X, y = args[0].X, args[0].y
        elif len(args) == 2:
            X, y = args
        if self.lib in ('sklearn', 'xgb'):
            self.model.fit(X, y)
            
    def predict(self, X):
        if self.lib in ('sklearn', 'xgb'):
            return self.model.predict_proba(X)[:, 1]
        
    def cross_validation(self, dataset, test_size=0.2, randomized=False, cat_idxs=None, **params):
        """
        Parameters depend on library
        sklearn:
        param_grid, scoring=None, 
        n_jobs=None, refit=True, cv=None, verbose
        
        xgb:
        nfold=3, num_boost_round=50,
        early_stopping_rounds=10, metrics="rmse"
        """
        X_train, y_train, X_test, y_test = dataset.get_data(test_size=test_size)
        if cat_idxs is None:
            cat_idxs = dataset.get_cat_idxs()
        else:
            cat_idxs = list(set(cat_idxs).union(dataset.get_cat_idxs()))
        
        if 'n_jobs' not in params:
            params['n_jobs'] = -1
        #params['estimator'] = self.model
        params['scoring'] = make_scorer(params['scoring'])
        
        if self.upsample:
            pipe_args = [
                ('sample', SMOTENC(categorical_features=cat_idxs, random_state=8)), 
                #('round', RoundData(round_columns)),
                ('classification', self.model),
            ]
            model = Pipeline(pipe_args)
        else:
            model = self.model
            
        if self.balance_test:
            X_test, y_test = SMOTENC(categorical_features=cat_idxs, random_state=8).fit_resample(X_test, y_test)
        #X_test, y_test = RoundData(round_columns).fit_transform(X_test, y_test)
        params['estimator'] = model
        self.X_test = X_test
        self.y_test = y_test
        
        if randomized:
            GridSearch = RandomizedSearchCV
            params['param_distributions'] = params['param_grid']
            del params['param_grid']
        else:
            GridSearch = GridSearchCV
        	
        cv = GridSearch(**params)
        cv.fit(X_train, y_train)
        self.cv = cv

        #elif self.lib == 'xgb':
            
        #    data_dmatrix = xgb.DMatrix(data=self.X, label=self.y)
            
        #    result = xgb.cv(dtrain=data_dmatrix, params=model.params,
        #                     as_pandas=True, seed=self.seed, **params)

    def predict(self, X, thresh=None):
        if thresh is None:
            return self.cv.best_estimator_.predict_proba(X)[:, 1]
        else:
            return np.where(self.cv.best_estimator_.predict_proba(X)[:, 1] > thresh, 1, 0)

        
    def get_result(self, metrics, thresh_list=[0.4, 0.5, 0.6], return_score=False):
        value_counts = self.y_test.value_counts().to_dict()
        most_freq = max(value_counts, key=value_counts.get)
        result = dict()
        result['analyze_data'] = {
            'best_estimator': self.cv.best_estimator_,
            'best_score': self.cv.best_score_,
            'best_params_': self.cv.best_params_,
            'class_balance': self.y_test[self.y_test == most_freq].shape[0] / self.y_test.shape[0]
        }
        for name, metric in metrics.items():
            for t in thresh_list:
                y_pred = self.predict(self.X_test, t)
                val = metric(self.y_test, y_pred)
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                result['analyze_data'][name + f'_{t}'] = val
    	    	
        if return_score:
            result['y_pred'] = self.predict(self.X_test)

        return result