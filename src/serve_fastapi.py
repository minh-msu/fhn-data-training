from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from test_schema import *

app = FastAPI(title="NYC Taxi Fare Prediction", version="1.0")
model = mlflow.pyfunc.load_model("models:/fare-model/Production")

@app.post("/predict")
def predict(item: Item):
    X = pd.DataFrame([item.dict()])
    y = model.predict(X)
    return {"prediction": float(y.item())}
