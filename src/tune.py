import mlflow
import mlflow.sklearn

import xgboost
from xgboost import XGBRegressor

import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from utils import *

def main():
    df = next(
        pd.read_csv("data/train.csv", chunksize=100000)
    )
    X, y = transform(df, "fare_amount")
    features = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        shuffle=False
    )
    cv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [100, 300, 500],          # number of boosting rounds
        "max_depth": [3, 5, 7, 9],                # tree depth
        "learning_rate": [0.01, 0.05, 0.1, 0.2]  # shrinkage step size
    }
    grid = GridSearchCV(
        estimator=XGBRegressor(), 
        param_grid=param_grid, 
        scoring="neg_root_mean_squared_error",
        cv=cv, 
        n_jobs=-1
    )
    
    mlflow.set_experiment("taxi-fare-prediction")
    mlflow.autolog(log_models=False)
    with mlflow.start_run(run_name="tuned_xgb"):
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        
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
