import pandas as pd
from utils import utils
from params import Params
from functools import cached_property
from typing import Dict, List, Any, Optional, Sequence, Union
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

class ModelRun(utils):

    def __init__(
        self,
        filepath : str,
        user_classifiers: List[str] = None,
        test_size: Optional[float]=0.3,
        grid_search: Optional[bool] = False
        ) -> None:

        self.filepath = filepath
        self.classifiers = user_classifiers if user_classifiers is not None else  __class__.classifiers
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
        return train_test_split(X, y, test_size=self.test_size)

    def scaled(self):
        features , y = self.data()
        scaler = StandardScaler()
        scaler.fit(features)
        X = scaler.transform(features)
        return train_test_split(X, y, test_size=self.test_size)

    def load(self) -> Dict:
        self.models={}
        for classifier in self.classifiers:
            if self.grid_search:
                params_grid = Params(self.classifiers[classifier]).param_grid()
                self.models[classifier] = GridSearchCV(self.classifiers[classifier](),params_grid,refit=True)
            else:
                self.models[classifier] = self.classifiers[classifier]()

    def execute(self):
        self.load()
        fit = {}
        predictions = {}
        
        for model in self.models:
            # Choose best K value
            if model == "K Nearest Neighbors":
                X_train, X_test, y_train, y_test = self.scaled()
            else:
                X_train, X_test, y_train, y_test = self.dataSplit()

            if self.grid_search:
                fit[model] = self.models[model].fit(X_train, y_train).best_estimator_
            else:
                fit[model] = self.models[model].fit(X_train, y_train)

            predictions[model] = fit[model].predict(X_test)
        return predictions, y_test

    def results(self) -> Dict[Dict]:
        results = {}
        predictions, y_test = self.execute()

        for model in predictions:
            results[model]={}
            results[model]['CR'] = classification_report(y_test, predictions[model])
            results[model]['CM'] = confusion_matrix(y_test, predictions[model])

        return results


##from sklearn.model_selection import cross_val_score
#cross_val_score()