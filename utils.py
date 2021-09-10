
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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

    available = [item for item in classifiers]
        
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
