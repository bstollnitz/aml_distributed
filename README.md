# How to build a distributed training application using PyTorch and Azure ML

This project shows how to build a distributed training application for a Fashion MNIST PyTorch model on Azure ML. It uses the Azure ML Python SDK API, and MLflow for tracking and model representation.


## Setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-83121-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-83121-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-83121-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-83121-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-83121-bstollnitz).
  * Install and activate the conda environment by executing the following commands:
  ```
  conda env create -f environment.yml
  conda activate aml_distributed
  ```
* In a terminal window, log in to Azure by executing `az login --use-device-code`. 
* Add a `config.json` file to the root of your project (or somewhere in the parent folder hierarchy) containing your Azure subscription ID, resource group, and workspace:
```
{
    "subscription_id": "<YOUR_SUBSCRIPTION_ID>",
    "resource_group": "<YOUR_RESOURCE_GROUP>",
    "workspace_name": "<YOUR_WORKSPACE>"
}
```
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-83121-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Install the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai), and log in to it by clicking on "Azure" in the left-hand menu, and then clicking on "Sign in to Azure."


## Train in the cloud

Select the run configuration "Train in the cloud" and press F5 to train in the cloud.

