# ml_pipeline

Machine learning pipeline using python and scikit learn for initial exploratory assessment

It does not cover data preparation nor feature engineering

## Usage

Data has been previously adjusted so the target is the last column

* Classifier ( 
    FilePath:str,
    GridSearch = False 
)

## Define the basic pipeline steps

User defined classifiers selection.

No user input: Can any assumption be made by data features (size, number or possible targets ...)

Hyper parameters Default x Grid search x Methods to manipulate (best K value etc.)


## Measuring Results + choosing the best model

Tabular model comparison (which metrics)
Quick visualizations

# TODOS 
* validate file format before read pandas
* check for text descritive feature, convert using pd get dummies method

### Is this information relevant ? 

 "Logistic Regression": 
 - A logistic regression is only suited to **linearly separable** problems
 - It's computationally fast and interpretable by design
 - It can handle non-linear datasets with appropriate feature engineering
 
 "Decision Tree": 
 - Decision trees are simple to understand and intrepret
 - They are prone to overfitting when they are deep (high variance)
 
 "Random Forest": 
 - They have lower risk of overfitting in comparison with decision trees
 - They are robust to outliers
 - They are computationally intensive on large datasets 
 - They are not easily interpretable
 
 "Gradient Boosting": 
 - Gradient boosting combines decision trees in an additive fashion from the start
 - Gradient boosting builds one tree at a time sequentially
 - Carefully tuned, gradient boosting can result in better performance than random forests
 
 "Neural Network": 
 - Neural Networks have great representational power but overfit on small datasets if not properly regularized
 - They have many parameters that require tweaking
 - They are computationally intensive on large datasets
 
 "K Nearest Neighbors": 
 - KNNs are intuitive and simple. They can also handle different metrics
 - KNNs don't build a model per se. They simply tag a new data based on the historical data
 - They become very slow as the dataset size grows
 
 "Gaussian Naive Bayes": 
 - The Naive Bayes algorithm is very fast
 - It works well with high-dimensional data such as text classification problems
 - The assumption that all the features are independent is not always respected in real-life applications
 
 "SVC":  
- SVMs or SVCs are effective when the number of features is larger than the number of samples
 - They provide different type of kernel functions
 - They require careful normalization   
    
