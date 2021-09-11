from utils import utils

import os
import numpy as np
import pandas as pd
from functools import cached_property
from typing import Dict, List, Any, Optional, Sequence, Union

from sklearn.metrics import  mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold, StratifiedKFold

#standarized features to avoid impact on the neighbors distance calculation
from sklearn.preprocessing import StandardScaler

class ModelLoad(utils):

    def __init__(
        self,
        user_classifiers: List[str] = None,
        user_params : Optional[Dict] = None

    ) -> None:
        self.user_classifiers = user_classifiers if user_classifiers is not None else  __class__.classifiers
        self.user_params = user_params
        # exec(__class__.imports[classifier] for classifier in self.user_classifiers)
        self.loadModels()

    def updateParams(self) -> dict:

        if self.user_params is not None:
            default = [k for k in __class__.default_parameters if k not in self.user_params]
            for param in default:
                self.user_params[param] =  __class__.default_parameters[param]
            return self.parameters
        else:
            return __class__.default_parameters
    
    def load(self) -> Dict:
        params = self.updateParams()
        models = {}
        for classifier in self.classifiers:
            models[classifier] = self.models[classifier](params[classifier])
        return models
                

class ModelRun(utils):

    def __init__(
        self,
        filepath : str,
        models : Dict[Union[str,function]],
        test_size: Optional[float]=0.3,
        grid_search: Optional[bool] = False
        ) -> None:

        self.models = models
        self.filepath = filepath
        self.test_size = test_size
        self.grid_search = grid_search

    @cached_property
    def data(self)-> pd.DataFrame:
        #fileName,fileExtension = os.path.splitext(self.filepath)
        df = pd.read_csv(self.filepath)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        return X , y

    def dataSplit(self)-> List[Union[Sequence , Any , list]]:
        X , y = self.data()
        return train_test_split(X, y, test_size=self.test_size, random_state=101)

    def scaled(self):
        features , y = self.data()
        scaler = StandardScaler()
        scaler.fit(features)
        X = scaler.transform(features)
        return train_test_split(X, y, test_size=self.test_size, random_state=101)

    # TODO implement Grid search
    # Choose best K value

    def execute(self):
        fit = {}
        predictions = {}
        X_train, X_test, y_train, y_test = self.dataSplit()

        for model in self.models:
            if model == "K Nearest Neighbors":
                 X_train, X_test, y_train, y_test = self.scaled()

            fit[model] = self.models[model].fit(X_train, y_train)
            predictions[model] = fit[model].predict(X_test)
            return predictions, y_test

    def results(self) -> Dict[Dict]:
        results = {}
        predictions, y_test = self.execute()

        for model in predictions:
            results[model]={}
            results[model]['MSE'] = np.sqrt(mean_squared_error(y_test, predictions[model]))
            results[model]['CR'] = classification_report(y_test, predictions[model])
            results[model]['CM'] = confusion_matrix(y_test, predictions[model])

        return results


#print(classification_report(y_test,predictions))

# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
# grid.fit(X_train,y_train)
# grid_predictions = grid.predict(X_test)
# best_classifier = grid.best_estimator_

# def scores (model, X_train, X_test, y_train, y_test):
#     model.fit(X_train,y_test)



##from sklearn.model_selection import cross_val_score

#cross_val_score()