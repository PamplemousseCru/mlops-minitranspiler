import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def build_linear_reg_model():
    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "linear_regression.joblib")

def build_logistic_reg_N_params_model():
    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = np.where(df["price"] < 0, -1, np.where(df["price"] > 270_000, 0, 1))
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "logistic_regression_N_params.joblib")

def build_logistic_reg_2_params_model():
    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = np.where(df["price"] > 270_000, 0, 1)
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "logistic_regression_2_params.joblib")

def build_decision_tree_model():
    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = np.where(df["price"] > 270_000, 0, 1)
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, "decision_tree.joblib")

build_linear_reg_model()
build_logistic_reg_N_params_model()
build_logistic_reg_2_params_model()
build_decision_tree_model()