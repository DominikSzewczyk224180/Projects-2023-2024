from azure.identity import ClientSecretCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup, Container, ResourceRequests, ResourceRequirements, 
    ContainerGroupNetworkProtocol, OperatingSystemTypes,
    IpAddress, Port
)

# Replace with your own values
SUBSCRIPTION_ID = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
RESOURCE_GROUP = 'buas-y2'
CONTAINER_NAME = "athena-backend"
IMAGE = 'kian183072/athena-backend:latest'  # Docker Hub image
CPU_CORE_COUNT = 1.0
MEMORY_GB = 1.5
TENANT_ID = "0a33589b-0036-4fe8-a829-3ed0926af886"
CLIENT_ID = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
CLIENT_SECRET = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

# Get credentials
credentials = ClientSecretCredential(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    tenant_id=TENANT_ID
)

# Create a Container Instance Management client
container_client = ContainerInstanceManagementClient(credentials, SUBSCRIPTION_ID)

# Define the container
container_resource_requests = ResourceRequests(memory_in_gb=MEMORY_GB, cpu=CPU_CORE_COUNT)
container_resource_requirements = ResourceRequirements(requests=container_resource_requests)
container = Container(name=CONTAINER_NAME, image=IMAGE, resources=container_resource_requirements, ports=[Port(port=80)])

# Define the group of containers
container_group = ContainerGroup(
    location='westeurope',
    containers=[container],
    os_type=OperatingSystemTypes.linux,
    ip_address=IpAddress(ports=[Port(protocol=ContainerGroupNetworkProtocol.tcp, port=80)], type='Public')
)

# Create the container group
container_client.container_groups.begin_create_or_update(RESOURCE_GROUP, CONTAINER_NAME, container_group)

print(f"Deployment of {CONTAINER_NAME} started.")
