# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.ai.ml.entities._deployment.resource_requirements_settings import (
    ResourceRequirementsSettings,
)
from azure.ai.ml.entities._deployment.container_resource_settings import (
    ResourceSettings,
)

import datetime
import time

subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV1"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

credential = ClientSecretCredential(tenant_id, client_id, client_secret)
# get a handle to the workspace
ml_client = MLClient(
    credential, subscription_id, resource_group, workspace_name
)

online_endpoint_name = "k8s-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = KubernetesOnlineEndpoint(
    name=online_endpoint_name,
    compute="adsai0",
    description="Cloud endpoint",   
    auth_mode="key",
)

ml_client.begin_create_or_update(endpoint).result()

env = Environment(
    conda_file="./Azure/example_conda.yml",
    image="mcr.microsoft.com/azureml/curated/tensorflow-2.16-cuda11:4"
)

registered_model_name = "U-net"
latest_model_version = 1
registered_environment_name = "athenamlenvironment5"
latest_environment_version = 13

model = ml_client.models.get(name=registered_model_name, version=latest_model_version)
# env = ml_client.environments.get(name=registered_environment_name, version=latest_environment_version)

blue_deployment = KubernetesOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./Azure/pipeline_scripts", scoring_script="scoring.py"
    ),
    instance_count=1,
    resources=ResourceRequirementsSettings(
        requests=ResourceSettings(
            cpu="16",
            memory="32Gi",
        ),
    ),
)

ml_client.begin_create_or_update(blue_deployment).result()

# blue deployment takes 100 traffic
# endpoint.traffic = {"blue": 100}
# ml_client.begin_create_or_update(endpoint).result()

# status = ml_client.online_endpoints.get(name=online_endpoint_name)

# print(status)

# logs = ml_client.online_deployments.get_logs(
#     name="blue", endpoint_name=online_endpoint_name, lines=50
# )

# print(logs)

# # Get the details for online endpoint
# endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# # existing traffic details
# print(endpoint.traffic)

# # Get the scoring URI
# print(endpoint.scoring_uri)

# blue deployment takes 100% traffic initially
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()

# Create green deployment
green_deployment = KubernetesOnlineDeployment(
    name="green",
    endpoint_name=online_endpoint_name,
    model=model,  # Update this if you have a new model version
    environment=env,  # Update this if you have a new environment version
    code_configuration=CodeConfiguration(
        code="./Azure/pipeline_scripts", scoring_script="scoring.py"
    ),
    instance_count=1,
    resources=ResourceRequirementsSettings(
        requests=ResourceSettings(
            cpu="16",
            memory="32Gi",
        ),
    ),
)

ml_client.begin_create_or_update(green_deployment).result()

# Set consistent traffic split
endpoint.traffic = {"blue": 90, "green": 10}  # 30% traffic to blue, 70% traffic to green
ml_client.begin_create_or_update(endpoint).result()

# Monitor green deployment and make adjustments if necessary
time.sleep(300)  # Wait for 5 minutes to monitor

# Check endpoint status and logs
status = ml_client.online_endpoints.get(name=online_endpoint_name)
print(status)

logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name=online_endpoint_name, lines=50
)
print(logs)

# Get the details for online endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)