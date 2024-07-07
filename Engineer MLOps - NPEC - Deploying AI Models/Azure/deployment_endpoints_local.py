import datetime
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (CodeConfiguration, ManagedOnlineDeployment, Model, Environment,
                                  ManagedOnlineEndpoint, KubernetesOnlineEndpoint, KubernetesOnlineDeployment)
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities._deployment.resource_requirements_settings import (
    ResourceRequirementsSettings,
)

from azure.ai.ml.entities._deployment.container_resource_settings import (
    ResourceSettings,
)

# Define your Azure ML settings
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV1"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

credential = ClientSecretCredential(tenant_id, client_id, client_secret)

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Define an endpoint name
# Example way to define a random name
local_endpoint_name = "local-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = KubernetesOnlineEndpoint(
    name=local_endpoint_name, description="this is a sample local endpoint"
)
print("Creating endpoint...")

ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)

# print(os.listdir('models'))
model = Model(path="models/None.h5")
env = Environment(
    conda_file="example_conda.yml",
    image="mcr.microsoft.com/azureml/curated/tensorflow-2.16-cuda11:5"#"deanis/azure-gpu-inference"#"mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)


blue_deployment = KubernetesOnlineDeployment(
    name="blue",
    endpoint_name=local_endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code=".", scoring_script="scoring.py"
    ),
    instance_count=1,
    resources=ResourceRequirementsSettings(
        requests=ResourceSettings(
            cpu="100m",
            memory="0.5Gi",
        ),
    ),
)

ml_client.online_deployments.begin_create_or_update(
    deployment=blue_deployment, local=True
)

status = ml_client.online_endpoints.get(name=local_endpoint_name, local=True)
print(status)

logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name=local_endpoint_name, local=True, lines=50
)

print(logs)

ml_client.online_endpoints.invoke(
    endpoint_name=local_endpoint_name,
    request_file='sample_data.json',
    local=True,
)
