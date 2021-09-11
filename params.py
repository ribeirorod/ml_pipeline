import re
import numpy as np
from  typing import Union
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin

class Params():

    """ Usage:
        params = Params(classifier=classifier).param_grid()
        apply params dict on grid search
        """

    # placeholders
    numParams = {'C' : np.linspace(0, 2, num=5),
                'gamma' : np.linspace(0, 1, num=5), }

    def __init__(
        self,
        classifier: Union[ClassifierMixin,RegressorMixin]) -> None:
        self.classifier = classifier


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
                params[k] = v.split('},')[0].strip().replace("'","").replace('{','').replace('"','')
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
        for param in num_params:
            params.update(__class__.numParams[param])

        return params

