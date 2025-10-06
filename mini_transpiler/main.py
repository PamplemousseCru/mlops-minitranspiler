import joblib
from builders.linear_regression import linear_regression_builder
from builders.logistic_regression import logistic_regression_builder
from builders.decision_tree import decision_tree_builder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

PATHS_TO_LOAD = ["linear_regression.joblib"]
PATHS_TO_LOAD += ["logistic_regression_N_params.joblib"]
PATHS_TO_LOAD += ["logistic_regression_2_params.joblib"]
PATHS_TO_LOAD += ["decision_tree.joblib"]


def load_models(paths) -> list:
    models = []
    for path in paths:
        model = joblib.load(path)
        models.append(model)
    return models

def write_to_file(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w+") as f:
        f.write(content)
    return path

def main():
    INPUTS = [205.9991686803,2,0]

    models = load_models(PATHS_TO_LOAD)

    for i, model in enumerate(models):
        print(f"=== MODEL {i} ===")
        if isinstance(model, LinearRegression):
            c_code = linear_regression_builder(model, INPUTS)
            write_to_file(Path("linear_regression.c"), c_code)
            print("The model has been transpiled to \"linear_regression.c\"")
            print(f"The model in Python predicted \"{model.predict([INPUTS])[0]}\", run the C file to check if it is the same")
        elif isinstance(model, LogisticRegression):
            c_code = logistic_regression_builder(model, [205.9991686803,2,0])
            if len(model.coef_) == 1:
                write_to_file(Path("logistic_regression_2_params.c"), c_code)
                print("The model has been transpiled to \"logistic_regression_2_params.c\"")
            else:
                write_to_file(Path("logistic_regression_N_params.c"), c_code)
                print("The model has been transpiled to \"logistic_regression_N_params.c\"")
            print(f"The model in Python predicted \"{model.predict([INPUTS])[0]}\", run the C file to check if it is the same")
        elif isinstance(model, DecisionTreeClassifier):
            c_code = decision_tree_builder(model, INPUTS)
            write_to_file(Path("decision_tree.c"), c_code)
            print("The model has been transpiled to \"decision_tree.c\"")
            print(f"The model in Python predicted \"{model.predict([INPUTS])[0]}\", run the C file to check if it is the same")
        else:
            continue

    return 0


if __name__ == "__main__":
    main()
