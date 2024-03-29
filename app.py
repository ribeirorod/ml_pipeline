#!/usr/bin/env

from Supervised.load import Model
import seaborn as sns

iris = sns.load_dataset('iris')
iris.to_csv('data.csv')

path= "data.csv"
classifiers = None
# size = default 0.3
# search = False

Client = Model(
        filepath = path, 
        test_size = 0.3, 
        grid_search=False)

CR, CM = Client.results()


# #Client.plot()
# import pandas as pd

# R = pd.DataFrame()