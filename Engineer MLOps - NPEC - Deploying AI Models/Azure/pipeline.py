from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.identity import ClientSecretCredential
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import os

subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV1"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

service_principal = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret,
)

workspace = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    auth=service_principal,
)

credential = ClientSecretCredential(tenant_id, client_id, client_secret)

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

environments = ml_client.environments.list()
print("Environments:\n")
for environment in environments:
    print(
        environment.name,
        ":",
    )
print("-------------------")

# List all available datasets
datasets = ml_client.data.list()
print("Datasets:\n")
for dataset in datasets:
    print(dataset.name, ":", dataset.latest_version)
print("-------------------")

# List all available compute targets
compute_targets = ml_client.compute.list()
print("Compute targets:\n")
for compute_target in compute_targets:
    print(compute_target.name, "-", compute_target.type)
print("-------------------")


environment_name = "athenamlenvironment5"
environment_version = 13
print(f"Using environment {environment_name} version {environment_version}")
compute_target_name = "adsai0"
print(os.listdir())

component_path = "./Azure/pipeline_scripts"
env = ml_client.environments.get(environment_name, environment_version)


train_component = command(
    name="train",
    display_name="Train model",
    description="Train model with data from a predefined data asset",
    inputs={
        "train_image_path": Input(type="uri_folder", description="Preprocessed data"),
        "train_mask_path": Input(type="uri_folder", description="Preprocessed data"),
        "val_image_path": Input(type="uri_folder", description="Preprocessed data"),
        "val_mask_path": Input(type="uri_folder", description="Preprocessed data"),
        'early_stopping': Input(type='boolean', description='Whether to include early stopping'),
        'epochs': Input(type='integer', description='Number of epochs for the model'),
        'hyperparameter_tuning': Input(type='boolean', description='Whether to perform hyperparameter tuning'),
        'save_model': Input(type='boolean', description='Whether to save the model')
    },
    outputs=dict(model=Output(type="uri_folder", mode="rw_mount")),
    code=component_path,
    command="python model_training_2.py --use_uri --train_image_path ${{inputs.train_image_path}} --train_mask_path ${{inputs.train_mask_path}} --val_image_path ${{inputs.val_image_path}} --val_mask_path ${{inputs.val_mask_path}} --model_save_path ${{outputs.model}} --early_stopping ${{inputs.early_stopping}} --n_epochs ${{inputs.epochs}} --hyperparameter_tuning ${{inputs.hyperparameter_tuning}} --save_model ${{inputs.save_model}}",
    environment=env,
    compute=compute_target_name
)

train_component = ml_client.create_or_update(train_component.component)

evaluate_component = command(
    name="evaluate",
    display_name="Evaluate model",
    description="Evaluate model with data from a predefined data asset",
    inputs={
        "model": Input(type="uri_folder", description="Model URI"),
        "test_images_path": Input(type="uri_folder", description="Path for the test data"),
        "test_masks_path": Input(type="uri_folder", description="Path for the test data"),
    },
    outputs=dict(
        accuracy_path=Output(type="uri_folder", description="Model accuracy output")
    ),
    code=component_path,
    command="python model_evaluation_temporary.py --use_uri --model_path ${{inputs.model}} --test_images_path ${{inputs.test_images_path}} --test_masks_path ${{inputs.test_masks_path}} --accuracy_path ${{outputs.accuracy_path}}",
    environment=env,
    compute_target=compute_target_name,
)

evaluate_component = ml_client.create_or_update(evaluate_component.component)

register_component = command(
    name="register",
    display_name="Register model",
    description="Register model with data from a predefined data asset",
    inputs={
        "model": Input(type="uri_folder", description="Model path for a model"),
        "accuracy": Input(type="uri_folder", description="Model accuracy file"),
    },
    code=component_path,
    command="python model_register.py --model ${{inputs.model}} --accuracy ${{inputs.accuracy}}",
    environment=env,
    compute_target=compute_target_name,
)

register_component = ml_client.create_or_update(register_component.component)


# list all components
components = ml_client.components.list()
print("Components:\n")
for component in components:
    print(component.name, ":", component.version)
print("-------------------")


@dsl.pipeline(
    name="ATHENA_pipeline_Kian",
    compute=compute_target_name,  # compute_target.name,
    # instance_type="gpu",
)
def train_eval_reg_pipeline(
    train_image: str,
    train_mask: str,
    val_image: str,
    val_mask: str,
    test_image: str,
    test_mask: str,
) -> None:
    training_step = train_component(train_image_path=train_image,
                                    train_mask_path=train_mask,
                                    val_image_path=val_image,
                                    val_mask_path=val_mask,
                                    early_stopping='yes',
                                    epochs=10,
                                    hyperparameter_tuning='no',
                                    save_model='yes')
    evaluation_step = evaluate_component(
        test_images_path = test_image,
        test_masks_path = test_mask,
        model=training_step.outputs.model
    )
    register_step = register_component(
        model=training_step.outputs.model, accuracy=evaluation_step.outputs.accuracy_path
    )


train_image = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV1/datastores/workspaceblobstore/paths/NPEC_Matey/train_images/"
)
train_mask = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV1/datastores/workspaceblobstore/paths/NPEC_Matey/train_masks/root/"
)
val_image = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV1/datastores/workspaceblobstore/paths/NPEC_Matey/val_images/"
)
val_mask = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV1/datastores/workspaceblobstore/paths/NPEC_Matey/val_masks/root/"
)
test_image = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV1/datastores/workspaceblobstore/paths/NPEC_Matey/test_images/"
)
test_mask = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV1/datastores/workspaceblobstore/paths/NPEC_Matey/test_masks/root/"
)
# Instantiate the pipeline.
pipeline_instance = train_eval_reg_pipeline(
    train_image=train_image,
    train_mask=train_mask,
    val_image=val_image,
    val_mask=val_mask,
    test_image=test_image,
    test_mask=test_mask,
)

# Submit the pipeline.
pipeline_run = ml_client.jobs.create_or_update(pipeline_instance)
