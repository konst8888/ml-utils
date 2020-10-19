from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np


class Classification:
    
    def __init__(self, lib, clf, seed=123, **params):
        if lib == 'xgb':
            default_params = {
                "objective":"reg:logistic",
                'colsample_bytree': 0.3,
                'learning_rate': 0.1,
                'max_depth': 5, 
                'alpha': 10
            }
            for p in default_params:
                if p not in params:
                    params[p] = default_params[p]
                    
        self.model = clf(**params)
        self.lib = lib
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
        
    def cross_validation(self, dataset, **params):
        """
        Parameters depend on library
        sklearn:
        param_grid, scoring=None, 
        n_jobs=None, refit=True, cv=None, verbose
        
        xgb:
        nfold=3, num_boost_round=50,
        early_stopping_rounds=10, metrics="rmse"
        """
        X_train, y_train, X_test, y_test = dataset.get_data(test_size=0.2)
        self.X_test = X_test
        self.y_test = y_test
        if self.lib == 'sklearn':
            if 'n_jobs' not in params:
                params['n_jobs'] = -1
            params['estimator'] = self.model
            params['scoring'] = make_scorer(params['scoring'], greater_is_better=True)
            	
            cv = GridSearchCV(**params)
            cv.fit(X_train, y_train)
            self.cv = cv

        elif self.lib == 'xgb':
            
            data_dmatrix = xgb.DMatrix(data=self.X, label=self.y)
            
            result = xgb.cv(dtrain=data_dmatrix, params=model.params,
                             as_pandas=True, seed=self.seed, **params)

        
    def get_result(self, metrics, thresh_list=[0.4, 0.5, 0.6]):
        result = {
            'best_estimator': self.cv.best_estimator_,
            'best_score': self.cv.best_score_,
            'best_params_': self.cv.best_params_,
            'class_balance': self.y_test[self.y_test == self.y_test.value_counts().argmax()].shape[0] / self.y_test.shape[0]
        }
        for name, metric in metrics.items():
            for t in thresh_list:
                val = metric(self.y_test, np.where(result['best_estimator'].predict_proba(self.X_test)[:, 1] > t, 1, 0))
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                result[name + f'_{t}'] = val
    	    	
        return result
