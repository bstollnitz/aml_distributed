"""Creates and runs an Azure ML command job."""

import logging
from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (AmlCompute, Data, Environment, Model)
from azure.ai.ml import PyTorchDistribution

from common import MODEL_NAME

COMPUTE_NAME = "cluster-distributed-gpu"
DATA_NAME = "data-fashion-mnist"
DATA_PATH = Path(Path(__file__).parent.parent, "data")
CONDA_PATH = Path(Path(__file__).parent, "conda.yml")
CODE_PATH = Path(Path(__file__).parent.parent, "src")
MODEL_PATH = Path(Path(__file__).parent.parent)
EXPERIMENT_NAME = "aml_distributed"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create the compute cluster.
    # Maximum of 2 nodes of Standard_NC24.
    # Each Standard_NC24 node has 4 NVIDIA Tesla K80 GPUs.
    cluster = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size="Standard_NC24",
        location="westus2",
        min_instances=0,
        max_instances=2,
    )
    ml_client.begin_create_or_update(cluster)

    # Create the data set.
    dataset = Data(
        name=DATA_NAME,
        description="Fashion MNIST data set",
        path=DATA_PATH.as_posix(),
        type=AssetTypes.URI_FOLDER,
    )
    ml_client.data.create_or_update(dataset)

    # Create the environment.
    environment = Environment(image="mcr.microsoft.com/azureml/" +
                              "openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:latest",
                              conda_file=CONDA_PATH)

    # Azure ML will set the MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE
    # environment variables on each node, in addition to the process-level RANK
    # and LOCAL_RANK environment variables, that are needed for distributed
    # PyTorch training.
    distr_config = PyTorchDistribution(process_count_per_instance=4)

    # Create the job.
    job = command(
        description="Trains a simple neural network on the Fashion-MNIST " +
        "dataset.",
        experiment_name=EXPERIMENT_NAME,
        compute=COMPUTE_NAME,
        inputs=dict(fashion_mnist=Input(path=f"{DATA_NAME}@latest")),
        outputs=dict(model=Output(type=AssetTypes.MLFLOW_MODEL)),
        code=CODE_PATH,
        environment=environment,
        distribution=distr_config,
        command="python train.py --data_dir ${{inputs.fashion_mnist}} " +
        "--model_dir ${{outputs.model}}",
    )
    job = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(job.name)

    # Create the model.
    model_path = f"azureml://jobs/{job.name}/outputs/model"
    model = Model(name=MODEL_NAME,
                  path=model_path,
                  type=AssetTypes.MLFLOW_MODEL)
    registered_model = ml_client.models.create_or_update(model)


if __name__ == "__main__":
    main()
