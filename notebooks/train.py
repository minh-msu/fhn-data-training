import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import xgboost
from xgboost import XGBRegressor

from transform import *

def main():
    df = next(
        pd.read_csv("train.csv", chunksize=100000)
    )

    X, y = transform(df, 'fare_amount')
    features = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        shuffle=False
    )

    client = mlflow.MlflowClient()
    mlflow.set_experiment("taxi-fare-prediction")
    mlflow.autolog(log_models=False)
    
    with mlflow.start_run(run_name="linreg-fare"):
        mlflow.set_tags({
            "algorithm": "linear_regression",
            "data_version": "25/2/2026", 
            "student_name": "minh_anh_dang",
        })

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_std, y_train)
        
        y_pred = model.predict(X_test_std).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_metric("rmse", rmse)

        importances = np.abs(model.coef_)
        with open("feature_importance.txt", "w") as f:
            for name, score in zip(features, importances):
                f.write(f"{name}: {score}\n")
        mlflow.log_artifact("feature_importance.txt")

        input_example=pd.DataFrame(
            X_train[:1], 
            columns=features
        )
        signature = infer_signature(
            X_train, 
            model.predict(X_train_std)
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model", 
            registered_model_name="fare-model",
            input_example=input_example, 
            signature=signature
        )

    version_N = client.get_latest_versions(
        name='fare-model', 
        stages=['None']
    )[0]
    metrics_N = client.get_run(version_N.run_id).data.metrics
    rmse_N = metrics_N.get("rmse", None)
    client.transition_model_version_stage(
        name="fare-model",
        version=version_N.version,
        stage="Staging",
    )
    print(f"Promote Version {version_N.version} to Staging")
    client.transition_model_version_stage(
        name="fare-model",
        version=version_N.version,
        stage="Production",
    )
    
    with mlflow.start_run(run_name="xgboost-fare"):
        mlflow.set_tags({
            "algorithm": "xgboost_regression",
            "data_version": "25/2/2026", 
            "student_name": "minh_anh_dang",
        })
            
        model = XGBRegressor(
            n_estimators=1000, 
            early_stopping_rounds=100
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )
        
        y_pred = model.predict(X_test).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_metric("rmse", rmse)

        importances = model.feature_importances_
        with open("feature_importance.txt", "w") as f:
            for name, score in zip(features, importances):
                f.write(f"{name}: {score}\n")
        mlflow.log_artifact("feature_importance.txt")
        
        signature = infer_signature(
            X_train, 
            model.predict(X_train)
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="fare-model",
            input_example=X_train.sample(n=1), 
            signature=signature
        )

    version_M = client.get_latest_versions(
        name='fare-model', 
        stages=['None']
    )[0]
    metrics_M = client.get_run(version_M.run_id).data.metrics
    rmse_M = metrics_M.get("rmse", None)
    if rmse_N and rmse_M and rmse_N>rmse_M:
        client.transition_model_version_stage(
            name="fare-model",
            version=version_M.version,
            stage="Production",
        )

    # Rollback
    # client.transition_model_version_stage(
    #         name="fare-model",
    #         version=version_M.version,
    #         stage="Staging",
    #         comment=f"Demoted to Staging"
    #     )
    # client.transition_model_version_stage(
    #         name="fare-model",
    #         version=version_N.version,
    #         stage="Staging",
    #         comment=f"Promoted to Production"
    #     )

if __name__ == "__main__":
    main()
