# Agreements
* Clearly communicate progress, and to do for the day if not able to attend stand-up

# Medal
> **Gold**: Create a web based GUI (using a frontend tool like REACT) for your deployed app that adds business value to the model.

> **Bronze**: Optional, if project is good, make blog post.

# Scrum master responsibilities
* Break down planning into days/weeks/phases and revise goals for weeks
* Prioritise tasks (must do - desirable, low priority - high priority)
* Time estimates per task
* Identify risks and how to eliminate/minimalise
* Plan in self-development hours
* Record all feedback given and upload to workspace + assign new tasks/action points
* Prepare and lead peer-review. Make sure everybody takes action and detail improvement points
* Lead meetings clearly and make sure we do not get side-tracked, plan another meeting if necessary
* Schedule meetings with product owner
* Review SMARTER goals per person in retrospective
* Ensure all tools/information is gathered to complete the sprint in product-owner meetings, prepare questions by looking forward to next week
* Motivate team
* Scrum master must ensure the right steps are taken to continue working next sprint
* Keep track of changes, announcements and notifications, discuss them in the morning stand-up

# Team
### Benjamin
* Strenghts: Azure, JavaScript
* Interest: Docker

### Daniel
* Strenghts: Model tuning, writing
* Interest: Continious learning

### Dominik
* Strenghts: Modelling, Streamlit
* Interest: Testing

### Kian
* Strenghts: Interface design, consumer perspective, business case, quality control
* Interest: API/packaging

### Matey
* Strenghts: Data assets/visualisations, modelling and tuning
* Interest: HTML/CSS/Interface, packaging


# Week planning
## Task Planning Template
#### Task description
* Person responsible
* Priority & necessity & effort (Low, Medium, Medium , High) (1-4)
* Time estimation & Deadline
* Relevant ILO (Poor = 1, Insufficient = 2, Sufficient = 3, Good = 4, Excellent = 5)
* Reviewer

#### Example: Creating week planning template
* Kian
* Medium priority 2 - Medium necessity 2 - Low effort 1
* 30 minutes - Deadline Friday 12:00
* Reviewed by team
* ILO: 1.1.1

## Week planning template - Focus
Goals:\
Relevant Requirements:

### Monday
* Stand-up - *(15 mins)*
* Sprint review - *(60 mins)*
* Sprint retrospective - *(60 mins)*
* Sprint planning in workspace & worklog - *(60 mins)*
* Group tasks - *(3 hrs)*
* Evidencing, logging, reflection - *(30 mins)*

### Tuesday
* Stand-up - *(15 mins)*
* Self study - *(3 hrs)*
* Individual tasks - *(4 hrs)*

### Wednesday
* Stand-up - *(15 mins)*
* Group tasks - *(6 hrs)*
* Product owner check-in - *(10 mins)*
* Evidencing, logging, reflection - *(30 mins)*

### Thursday
* Stand-up - *(15 mins)*
* Self study - *(2 hrs)*
* Individual tasks - *(3 hrs)*
* Extra self development - *(2 hrs)*

### Friday
* Stand-up - *(15 mins)*
* Self study - *(3 hrs)*
* Individual tasks - *(3 hrs)*
* Evidencing, logging, reflection - *(30 mins)*

## Week 1 - Scoping - Kian
> Code should be modular using functions + write unit-testing (coverage 95%)\
> On GitHub as .py\
> Inidicate contribution\
> Instructions to set up virtual environment + README.md\
> Ensure compatibility with CPU and GPU / system resources and Docker compatibility\
> Static type checking?\
> Clear documentation/explaination, type hinting (checking dtype), logging (print statements for debugging) and PEP8\
> Allow various options for the programme to be run\
> Modules contain output files where valuable


## Week 2 & 3 - Create Production Ready Code - Matey
*Daniel & Kian will be absent, drinking beers in Vienna*
> Code should be modular using functions + write unit-testing (coverage 95%)\
> On GitHub as .py\
> Inidicate contribution\
> Instructions to set up virtual environment + README.md\
> Ensure compatibility with CPU and GPU / system resources and Docker compatibility\
> Static type checking?\
> Clear documentation/explaination, type hinting (checking dtype), logging (print statements for debugging) and PEP8\
> Allow various options for the programme to be run\
> Modules contain output files where valuable

## Week 4 & 5 - Data Pipelines & Model Training in the Cloud - Dominik
*Daniel & Kian will be absent, drinking beers in Hungary in week 4*

> Be able to store data in cloud + manage through code\
> Be able to train locally/cloud\
> Track key metrics\
> Implement training pipelines/automated hyperparameter tuning (Optuna)/various model architectures + version control on cloud\
> Workflow automation + metric logging and visualisation\
> Monitor data drift\
> Training costs analysis\
> Design and plan cloud architecture and ensuring API works with real-time data

## Week 6 & 7 - Model Deployment & Monitoring / Testing & Evaluation - Benjamin
> Creating API works with real-time data and also accept bastches of labelled data \
> Implement advanced deployment strategies (blue/green deployment)\
> Deploy multiple models (including open-source) that can be swapped \
> Implement continous training/testing\
> Dev/Test/Prod environments / branches used as part of the process\
> Appropriate alerting systems in place\
> Usage and capacity is monitored\
> Test deployment using realistic traffic generation\
> Automate code update verification\
> Tracing is implemented to identify bottlenecks
> Unit testing + writing errors

## Week 8 - Finalising - Daniel
> Checking all requirements

# Deliverables
* A modular Python package that can be installed using pip or whl file.
* A fully-functional CLI for interacting with the application.
* A web interface/API for interacting with the application.
* Comprehensive documentation, including installation and usage instructions, troubleshooting tips, and example code.
* A containerised application that can be deployed on Azure services and local machines.
* A comprehensive demonstration of the application's functionality and MLOps best practices.

## Requirements
* The system should be modular and should be robotic platform agnostic.
* The system should accept an image as input and should output the segmentation masks and landmark locations.
* Information about the certainty of the predictions should also be provided.
* It should be possible to train the system on a local machine or on a cloud service - through the CLI and API.
* Inference should be performed on images from the Hades system and should be displayed on a web interface and be accessible through an API. This allows the system to be deployed on any robotic platform regardless of the hardware specifications.
* Incorporate MLOps best practices for managing data environments, training, and deployment on Azure Machine Learning in order to ensure the system is scalable and maintainable.

## General Requirements

* Develop a scalable API that accepts input data and returns predictions along with confidence scores
* Ensure that the model supports automatic retraining on a weekly basis or when new data is available
* Continuously monitor prediction accuracy to prevent model drift
* Maintain clear records of the deployed model and its training process for auditing purposes (data, code, and model versioning)
* Minimize manual intervention in deployment, with manual approval required for production (e.g., via a CI/CD pipeline, automated testing, and deployment)
* Store and manage code in a code repository, such as GitHub
## Technical requirements
### Python Package
* Combine the PoC notebooks and scripts into a single, cohesive Python package.
* Organise the package into modular components for easy maintenance and extensibility.
* Ensure the package adheres to Python best practices and follows PEP8 guidelines.
* Use logging to record the application's progress and output, and to provide useful information for debugging.
* Implement a user-friendly CLI for interacting with the application.
* Support basic functionality such as training, evaluation, and inference through the CLI.
* Provide options for customization of input/output file paths, model parameters, and other relevant settings.
* Unit test the package to ensure its functionality and reliability.
### Documentation
* Use docstrings and type hints to document the Python package.
* Create comprehensive documentation for the Python package, CLI, API, and MLOps processes.
* Include detailed instructions for installation, usage, and troubleshooting.
* Provide examples and sample code to demonstrate the application's functionality and MLOps best practices.
* Ideally the documentation should be hosted on a website using GitHub Pages.
### MLOps Best Practices
* Set up data environments on Azure, ensuring proper data storage, versioning, and access control.
* Implement automated training pipelines with Azure Machine Learning, incorporating hyperparameter tuning and model evaluation.
* Integrate CI/CD practices for seamless deployment and updates of the application.
* Monitor the performance of the deployed models, implementing automated alerts and retraining when necessary.
* Maintain detailed logs of model experiments, training runs, and deployments.
* Implement a rollback strategy in case of unexpected errors or performance degradation.
* Allow for continuous model training and deployment using the latest data with minimal human intervention and sound deployment practices and strategies.
### API Deployment
* Develop an API for deploying the containerised application on Azure services and local machines.
* Ensure proper authentication and security measures are in place.
* Routes for training, evaluation, and inference should be implemented within the API.
* Ideally the API should be accessible through a web interface and should also be callable from other applications.
### Testing and Validation
* Test the application on a range of scenarios to ensure its accuracy, reliability and scalability.
* Validate the application's performance using relevant evaluation metrics.
* Ensure the application is robust to common errors and exceptions.