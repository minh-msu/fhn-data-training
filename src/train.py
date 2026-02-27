import mlflow
import mlflow.sklearn

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import xgboost
from xgboost import XGBRegressor

from utils import *

def main():
    df = next(
        pd.read_csv("../data/train.csv", chunksize=100000)
    )

    X, y = transform(df, 'fare_amount')
    features = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        shuffle=False
    )

    mlflow.set_experiment("taxi-fare-prediction")
    mlflow.autolog(log_models=False)
    
    with mlflow.start_run(run_name="linreg-fare"):
        mlflow.set_tags({
            "algorithm": "linear_regression",
            "data_version": "25/2/2026", 
            "student_name": "minh_anh_dang",
        })

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test).ravel()
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)

        importances = np.abs(model.coef_)
        with open("feature_importance.txt", "w") as f:
            for name, score in zip(features, importances):
                f.write(f"{name}: {score}\n")
        mlflow.log_artifact("feature_importance.txt")

        mlflow.sklearn.log_model(
            model,
            name="model", 
            registered_model_name="fare-model",
            input_example=X_train.sample(n=1)
        )
    
    with mlflow.start_run(run_name="xgboost-fare"):
        mlflow.set_tags({
            "algorithm": "xgboost_regression",
            "data_version": "25/2/2026", 
            "student_name": "minh_anh_dang",
        })
      
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test).ravel()
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)

        importances = model.feature_importances_
        with open("feature_importance.txt", "w") as f:
            for name, score in zip(features, importances):
                f.write(f"{name}: {score}\n")
        mlflow.log_artifact("feature_importance.txt")
        
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="fare-model",
            input_example=X_train.sample(n=1)
        )

if __name__ == "__main__":
    main()
