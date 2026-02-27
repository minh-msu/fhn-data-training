import argparse
import pandas as pd
import mlflow.pyfunc
from utils import *


def batch_infer(model_uri: str, input_path: str, output_path: str):
    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.read_csv(input_path)
    X = transform(df)
    predictions = model.predict(X)

    df["prediction"] = predictions
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script")

    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="MLflow model URI (e.g. models:/fare-model/Production)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file",
    )

    args = parser.parse_args()

    batch_infer(
        model_uri=args.model_uri,
        input_path=args.input,
        output_path=args.output,
    )