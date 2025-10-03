def build_model():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression

    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    # y = df["price"]
    y = np.where(df["price"] < 0, -1, np.where(df["price"] > 270_000, 0, 1))
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "linear_regression.joblib")


build_model()
