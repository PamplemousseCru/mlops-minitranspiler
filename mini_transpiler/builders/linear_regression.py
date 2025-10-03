from builders.utils import get_c_list, get_imports
from functions.utils import linear_regression
from sklearn.linear_model import LinearRegression


def _main(features, thetas) -> str:
    return f"""
int main(int argc, char** argv) {{
    double features[{len(features)}] = {features};
    double thetas[{len(thetas)}] = {thetas};
    double prediction = linear_regression_prediction(features, thetas, {len(thetas)});
    printf("Prediction: %f\\n", prediction);
    return 0;
}}
"""


def linear_regression_builder(model: LinearRegression, features) -> str:
    thetas = [model.intercept_] + model.coef_.tolist()
    thetas = get_c_list(thetas)
    features = get_c_list(features)
    n_parameters = len(thetas)  # Including intercept
    return f"""
{get_imports()}
{linear_regression()}
{_main(features, thetas)}
"""
