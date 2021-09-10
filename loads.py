
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Sequence, Union


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

#standarized features to avoid impact on the neighbors distance calculation
from sklearn.preprocessing import StandardScaler

class utils():

    imports = {
    "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
    "Decision Tree": "from sklearn.tree import DecisionTreeClassifier",
    "Random Forest": "from sklearn.ensemble import RandomForestClassifier",
    "Gradient Boosting": "from sklearn.ensemble import GradientBoostingClassifier",
    "Neural Network": "from sklearn.neural_network import MLPClassifier",
    "K Nearest Neighbors": "from sklearn.neighbors import KNeighborsClassifier",
    "Gaussian Naive Bayes": "from sklearn.naive_bayes import GaussianNB",
    "SVC": "from sklearn.svm import SVC",
    }

    # To be pulled by the users in order to change only specific parameters 
    default_parameters = {
        "Logistic Regression": {
            "random_state": 101,
            "solver": "lbfgs", 
            "penalty": "l2", 
            "C": 1.0, 
            "max_iter": 100},

        "Decision Tree":{
            "random_state": 101,
            "solver": "lbfgs", 
            "penalty": "l2", 
            "C": 1.0, 
            "max_iter": 100}
            }
 
    classifiers = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Neural Network": MLPClassifier,
    "K Nearest Neighbors": KNeighborsClassifier,
    "Gaussian Naive Bayes": GaussianNB,
    "SVC": SVC,
    }
            
    urls = {
    "Logistic Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "Decision Tree": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "Random Forest": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "Gradient Boosting": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "Neural Network": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
    "K Nearest Neighbors": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
    "Gaussian Naive Bayes": "https://scikit-learn.org/stable/modules/naive_bayes.html",
    "SVC": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
    }

    infos = {
        "Logistic Regression": """
        - A logistic regression is only suited to **linearly separable** problems
        - It's computationally fast and interpretable by design
        - It can handle non-linear datasets with appropriate feature engineering
        """,
        "Decision Tree": """
        - Decision trees are simple to understand and intrepret
        - They are prone to overfitting when they are deep (high variance)
        """,
        "Random Forest": """
        - They have lower risk of overfitting in comparison with decision trees
        - They are robust to outliers
        - They are computationally intensive on large datasets 
        - They are not easily interpretable
        """,
        "Gradient Boosting": """
        - Gradient boosting combines decision trees in an additive fashion from the start
        - Gradient boosting builds one tree at a time sequentially
        - Carefully tuned, gradient boosting can result in better performance than random forests
        """,
        "Neural Network": """
        - Neural Networks have great representational power but overfit on small datasets if not properly regularized
        - They have many parameters that require tweaking
        - They are computationally intensive on large datasets
        """,
        "K Nearest Neighbors": """
        - KNNs are intuitive and simple. They can also handle different metrics
        - KNNs don't build a model per se. They simply tag a new data based on the historical data
        - They become very slow as the dataset size grows
        """,
        "Gaussian Naive Bayes": """
        - The Naive Bayes algorithm is very fast
        - It works well with high-dimensional data such as text classification problems
        - The assumption that all the features are independent is not always respected in real-life applications
        """,
        "SVC": """
        - SVMs or SVCs are effective when the number of features is larger than the number of samples
        - They provide different type of kernel functions
        - They require careful normalization   
        """,
        }


class loadModels(utils):

    """ 
    Loads a range of pre defined classifiers on scikit learn library
    algorithms : List 
    parameters : Optional [Dict]
    """
    def __init__(
        self,
        user_classifiers: List[str] = None,
        user_params : Optional[Dict] = None

    ) -> None:

        self.user_classifiers = user_classifiers if user_classifiers is not None else  __class__.classifiers
        self.user_params = user_params
        exec(__class__.imports[classifier] for classifier in self.user_classifiers)
        self.loadModels()

    def updateParams(self) -> dict:

        if self.user_params is not None:
            default = [k for k in __class__.default_parameters.keys() not in self.user_params]
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

    # TODO implement grid search
    # Choose K value

    def run(self):
        fit = {}
        predictions = {}
        X_train, X_test, y_train, y_test = self.data()

        for model in self.models:
            if model == "K Nearest Neighbors":
                 X_train, X_test, y_train, y_test = self.scaled_data()

            fit[model] = self.models[model].fit(X_train, y_train)
            predictions[model] = fit[model].predict(X_test)
            return predictions, y_test

    def metrics(self):
        results = {}
        predictions, y_test = self.run()

        for model in predictions:
            results[model]={}
            results[model]['MSE'] = np.sqrt(mean_squared_error(y_test, predictions[model]))
            results[model]['CR'] = classification_report(y_test, predictions[model])
            results[model]['CM'] = classification_report(y_test, predictions[model])
