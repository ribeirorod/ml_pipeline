import pandas as pd
import numpy as np
import sys 
from .classifiers import Classifiers
from .params import Params
from functools import cached_property
from typing import Dict, List, Any, Optional, Sequence, Union
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

class Model(Classifiers):

    def __init__(
        self,
        filepath : str,
        user_classifiers: List[str] = None,
        test_size: Optional[float]=0.3,
        grid_search: Optional[bool] = False
        ) -> None:

        self.filepath = filepath
        self.classifiers = user_classifiers if user_classifiers is not None else  __class__.models
        self.test_size = test_size
        self.grid_search = grid_search

    @cached_property
    def data(self) -> List[Union[Sequence , Any , list]]:
        #fileName,fileExtension = os.path.splitext(self.filepath)
        df = pd.read_csv(self.filepath).reset_index()
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        return train_test_split(X, y, test_size=self.test_size)

    def load_grid(self) -> List:
        params_grid = {}
        for i, (name, model) in enumerate(self.models):
            params_grid = Params(name, model).param_grid()
            self.models[i] = (name, GridSearchCV(model,params_grid,refit=True))

    @cached_property
    def execute(self) -> tuple:

        self.load_grid() if self.grid_search else None
        fit = {}
        predictions = {} 
        exceptions =[]  
        X_train, X_test, y_train, y_test = self.data

        for (name, model) in self.models:
            if self.grid_search:
                fit[name] = model.fit(X_train, y_train).best_estimator_
            else:
                try:
                    fit[name] = model.fit(X_train, y_train)
                except:
                    print(f'Error fitting {name}: ', sys.exc_info()[0])
                    continue

            try:
                predictions[name] = fit[name].predict(X_test)
                print(name,': fit and predict - OK' )
            except:
                pass
        return (predictions, y_test)

    def results(self) -> Dict:
        CR , CM = dict(), dict()
        predictions, y_test = self.execute

        for model in predictions:
            CR[model] = classification_report(y_test, predictions[model])
            CM[model] = confusion_matrix(y_test, predictions[model])

        return CR, CM

    def plot(self, weights=20):
        predictions , yt = self.execute

        for name in predictions:
            figure, axe_1 = plt.subplots(1, 1, figsize=(14,5))
            p = predictions[name]
            label = f' Predicted - {name}'        
        ##
            axe_1.plot   (yt, yt, color='g', label="Ground Truth")
            axe_1.scatter(yt, p , color='r', label=label , alpha=0.25, s=weights)
            axe_1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            axe_1.set_ylabel(label)
            axe_1.set_xlabel("Ground Truth")
            axe_1.axhline(0, color="black")
            axe_1.axvline(0, color="black")
        # ##
        #     error = 0 if yt == p else 1
        #     mu    = np.mean(error)
        #     sigma = np.std(error)
        # ##
        #     axe_2.hist( error, int((error.shape[0]*1.0)/2), normed=True, facecolor='r', alpha=0.15)
        #     axe_2.set_title(r'Error distribution ($\mu=%.3f,\ \sigma=%.3f$)'%(mu,sigma))
        ##
            plt.show()