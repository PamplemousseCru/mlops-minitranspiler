from builders.utils import get_c_list, get_imports
from functions.utils import linear_regression
from sklearn.linear_model import LogisticRegression


def linear_regression_builder(model: LogisticRegression, features) -> str:
    if len(model.coef_) == 1:
        # only two classes so we call then sigmoid and we get probability of class 1
        # so 1 - that and proba of class 0
        pass
    else:
        # add intercepts to evey sub list then call the get_c_matrix
        # then call linear_regression on each and then argmax the result on classes
        pass
