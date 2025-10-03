import joblib
from builders.linear_regression import linear_regression_builder
from sklearn.linear_model import LinearRegression, LogisticRegression

PATHS_TO_LOAD = ["linear_regression.joblib"]


def load_models(paths) -> list:
    models = []
    for path in paths:
        model = joblib.load(path)
        models.append(model)
    return models


def main():
    models = load_models(PATHS_TO_LOAD)

    for model in models:
        if isinstance(model, LinearRegression):
            print(linear_regression_builder(model, [2000, 4, 1]))
        elif isinstance(model, LogisticRegression):
            pass

    return 0


if __name__ == "__main__":
    main()
