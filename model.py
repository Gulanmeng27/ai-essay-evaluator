import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r, p = pearsonr(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "pearson_r": round(r, 4),
        "p_value": round(p, 6),
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "predictions": y_pred.tolist(),
    }


def get_feature_importance(model, feature_names):
    coefs = list(zip(feature_names, model.coef_))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    return coefs


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
