
import re
import numpy as np
from  typing import Union
from sklearn.base import RegressorMixin ,ClassifierMixin
from sklearn.utils.fixes import loguniform
from collections.abc import Iterable

class Params():

    # Numeric hyperparameters: range placeholders
    numParams = {'C' : loguniform.rvs(1e0, 1e2, size=5),
                'gamma' : loguniform.rvs(1e-4, 1e-3, size=5),
                'degree': np.linspace(1, 5, num=2),
                'base_estimator__max_depth': [2, 4, 6, 8]}

    def __init__(
        self,
        cl_name : str,
        classifier: Union[ClassifierMixin,RegressorMixin]) -> None:
        self.cl_name = cl_name
        self.classifier = classifier

    def exceptions(self, params) -> dict:
        for k in ['bootstrap', 'class_weight']:
            params.pop(k) if k in params.keys() else None

        if self.cl_name == "Logistic Regression":
            params['dual'] = [False]
        return params

    def param_grid(self) -> dict:
        params={}
        
        text = str(self.classifier.__doc__)
        ubound , lbound = ['    Parameters\n', '    Attributes\n']
        start , end = text.find(ubound), text.find(lbound)
        params_text = text[start:end].splitlines()
        
        for line in params_text:
            if re.findall(pattern='\s:\s\{', string=line):
                k , v = line.split(':')
                k = k.strip()
                params[k] = {}
                params[k] = v.split('}')[0].strip().replace("'","").replace('{','').replace('"','')
                params[k] = [x.strip() for x in params[k].split(',')]

            elif re.findall(pattern='\s:\s', string=line):
                k , v = line.split(':')
                k = k.strip()
                params[k] = v.split(',')[0].strip()

                if params[k] == 'bool':
                    params[k] = [True, False]
                elif any( substr in 'int float' for substr in params[k]):
                    params.pop(k)

        all_params = self.classifier._get_param_names()
        num_params = [param for param in __class__.numParams if param in all_params]

        if isinstance(num_params, Iterable):
            for param in num_params:
                params.update(param = __class__.numParams[param])
        elif len(num_params) == 1:
             params.update(num_params=__class__.numParams[num_params])

        return self.exceptions(params)

