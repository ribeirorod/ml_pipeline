
from sklearn.pipeline      import Pipeline
from sklearn               import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler


class Classifiers():

    # variaveis numericas quantitativas x qualitativas (scalar standarization nao se aplica)
    scaler = ('scl', StandardScaler())

    models = [ ('Lasso'                      , linear_model.Lasso(alpha = 0.1))
                  , ('Lasso LARS'            , linear_model.Lars(n_nonzero_coefs=1))
                  , ('OMP'                   , linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=1))
                  , ('Linear Regression'     , linear_model.LinearRegression() )
                  , ('Ridge Regression'      , linear_model.Ridge(alpha=0.1))
                  , ('Bayesian Ridge'        , linear_model.BayesianRidge())
                  , ('SVR'                   , Pipeline([ scaler,('svr', SVR(C=10.0, kernel="linear"))
                                                        ])
                    )
                  , ('Auto Relevance'        , linear_model.ARDRegression())
                  , ("Logistic Regression"   , linear_model.LogisticRegression)
                  , ("Decision Tree"         , DecisionTreeClassifier())
                  , ("Random Forest"         , RandomForestClassifier())
                  , ("Gradient Boosting"     , GradientBoostingClassifier())
                  , ("Neural Network"        , MLPClassifier())
                  , ("K Nearest Neighbors"   , Pipeline([ scaler, ('knn', KNeighborsClassifier())
                                                        ])
                    )
                  , ("Gaussian Naive Bayes"  , GaussianNB())
                  , ("SVC"                   , SVC()),
    ]

    available = [name for (name, model) in models]
        
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


