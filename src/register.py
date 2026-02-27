import mlflow
import argparse

def transition(model_name, version, stage, rollback=False):
    client = mlflow.MlflowClient()
    if rollback:
        prod = client.get_latest_versions(
            name=model_name, 
            stages=["Production"]
        )[0]
        client.transition_model_version_stage(
                name=model_name,
                version=prod.version,
                stage="Staging"
            )
        print(f"Demoted Production model to Staging")
        
    client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    print(f"Promote {model_name} version {version} to {stage}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Registry script")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of MLFlow model",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version whose stage will be transitioned",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Stage that the model will be transitioned into",
    )
    parser.add_argument(
        "--rollback",
    	action="store_true",
        help="Rollback the production model",
    )

    args = parser.parse_args()

    transition(
        model_name=args.model_name,
        version=args.version,
        stage=args.stage,
        rollback=args.rollback
    )