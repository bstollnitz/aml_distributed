"""Creates and runs an Azure ML command job."""

import logging
from pathlib import Path

from azure.ai.ml import MLClient, Output, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model
from azure.identity import DefaultAzureCredential

COMPUTE_NAME = "cluster-distributed-gpu"
DATA_NAME = "data-fashion-mnist"
DATA_PATH = Path(Path(__file__).parent.parent, "data")
CONDA_PATH = Path(Path(__file__).parent, "conda.yml")
CODE_PATH = Path(Path(__file__).parent.parent, "src")
MODEL_PATH = Path(Path(__file__).parent.parent)
EXPERIMENT_NAME = "aml_distributed"
MODEL_NAME = "model-distributed"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create the compute cluster.
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

    # Create the environment.
    environment = "AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu@latest"
    # environment = Environment(image="mcr.microsoft.com/azureml/" +
    #                           "openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:latest",
    #                           conda_file=CONDA_PATH)

    # Notice that we specify that we want two nodes/instances, and 4 processes
    # per node/instance.
    # 2 instances * 4 processes per instance = 8 total processes.
    # Azure ML will set the MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE
    # environment variables on each node, in addition to the process-level RANK
    # and LOCAL_RANK environment variables, that are needed for distributed
    # PyTorch training.
    job = command(
        description="Trains a simple neural network on the Fashion-MNIST " +
        "dataset.",
        experiment_name=EXPERIMENT_NAME,
        compute=COMPUTE_NAME,
        outputs=dict(model=Output(type=AssetTypes.MLFLOW_MODEL)),
        code=CODE_PATH,
        environment=environment,
        resources=dict(instance_count=2),
        distribution=dict(type="PyTorch", process_count_per_instance=4),
        command="python train.py " + "--model_dir ${{outputs.model}}",
    )
    job = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(job.name)

    # Create the model.
    model_path = f"azureml://jobs/{job.name}/outputs/model"
    model = Model(name=MODEL_NAME,
                  path=model_path,
                  type=AssetTypes.MLFLOW_MODEL)
    ml_client.models.create_or_update(model)


if __name__ == "__main__":
    main()
