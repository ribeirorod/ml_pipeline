from utils import utils

import os
import numpy as np
import pandas as pd

from typing import Dict, List, Any, Optional, Sequence, Union

from sklearn.metrics import  mean_squared_error, explained_variance_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

#standarized features to avoid impact on the neighbors distance calculation
from sklearn.preprocessing import StandardScaler

class loadModels(utils):

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
    
    def loadModels(self) -> Dict:
        params = self.updateParams()
        models = {}
        for classifier in self.classifiers:
            models[classifier] = self.models[classifier](params[classifier])
        return models
                

class RunModels(utils):

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

    def data(self)-> List[Union[Sequence , Any , list]]:
        # TODO validate file format 
        fileName,fileExtension = os.path.splitext(self.filepath)
        
        df = pd.read_csv(self.filepath)
        #convention to always put target column at last
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=101)
        return X_train, X_test, y_train, y_test

    def scaled_data(self):
        df = pd.read_csv(self.filepath)
        scaler = StandardScaler()
        scaler.fit(df.drop(df.columns[:-1], axis=1))
        X = scaler.transform(df.drop(df.columns[:-1], axis=1))
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=101)
        return X_train, X_test, y_train, y_test

    # TODO implement Grid search
    # Choose best K value

    def execute(self):
        fit = {}
        predictions = {}
        X_train, X_test, y_train, y_test = self.data()

        for model in self.models:
            if model == "K Nearest Neighbors":
                 X_train, X_test, y_train, y_test = self.scaled_data()

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
            results[model]['CM'] = classification_report(y_test, predictions[model])

        return results


#print(classification_report(y_test,predictions))

# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
# grid.fit(X_train,y_train)
# grid_predictions = grid.predict(X_test)