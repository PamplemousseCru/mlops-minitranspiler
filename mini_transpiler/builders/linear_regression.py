from builders.utils import get_c_list, get_imports
from functions.utils import linear_regression
from sklearn.linear_model import LinearRegression


def _main(features, thetas, n_features, n_thetas) -> str:
    return f"""
int main(int argc, char** argv) {{
    double features[{n_features}] = {features};
    double thetas[{n_thetas}] = {thetas};
    double prediction = linear_regression_prediction(features, thetas, {n_thetas});
    printf("Prediction: %f\\n", prediction);
    return 0;
}}
"""


def linear_regression_builder(model: LinearRegression, features) -> str:
    n_features = len(features)
    n_thetas = len(model.coef_) + 1 # + intercept

    thetas = [model.intercept_] + model.coef_.tolist()
    thetas = get_c_list(thetas)
    features = get_c_list(features)
    return f"""{get_imports()}
{linear_regression()}
{_main(features, thetas, n_features, n_thetas)}
"""
