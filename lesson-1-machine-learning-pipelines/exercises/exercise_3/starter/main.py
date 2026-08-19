import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        },
    )

    ##################
    # Your code here: use the artifact we created in the previous step as input for the `process_data` step
    # and produce a new artifact called "cleaned_data".
    # NOTE: use os.path.join(root_path, "process_data") to get the path
    # to the "process_data" component
    ##################

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "input_artifact": "iris.csv:latest",
            "artifact_name": "cleaned_data.csv",
            "artifact_type": "cleaned_data",
            "artifact_description": "Cleaned data"
        },
    )


if __name__ == "__main__":
    go()


"""
It wouldn't run it couldn't find the version in the mlflow.db. In short I had to activate
new conda env then run the project from there. super weird.


# navigate to the folder
source ~/miniconda3/activate/source
conda activate mledp2

# remove the stale MLflow DB
rm -f mlflow.db

# optional: remove old outputs if needed
rm -rf outputs

# activate the env where mlflow is installed
conda activate mlflow-42967bdede9d0e24b9a67b0617f50feab61dfa9a

# verify
which mlflow
mlflow --version

# run the project
export HYDRA_FULL_ERROR=1
mlflow run . -P hydra_options="main.experiment_name=prod"
"""