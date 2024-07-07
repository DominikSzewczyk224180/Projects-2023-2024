# MLOps Plan

### 1. GitHub Version Control and Branching
- **Branching Strategy:**
  - **Development Branch:** All new features and updates are implemented here.
  - **Testing Branch:** The development branch is merged into the test branch for automated testing.
  - **Main Branch:** Upon passing tests, the test branch is automatically merged into the main branch.
  
- **Automated Testing and Documentation:**
  - Unit tests run automatically when changes are pushed to the test branch.
  - Code coverage reports and unit test results are generated.
  - Sphinx documentation is built and published on GitHub Pages.

### 2. Docker Containerization
- **Docker Containers:**
  - Use Docker for CLI, frontend, and backend services.
  - Automate the build process for Docker images:
    - **CLI Docker Image:** Tools and scripts for interacting with Azure ML and other services.
    - **Frontend Docker Image:** UI components of the application.
    - **Backend Docker Image:** API services, including FastAPI.
  - Ensure all Docker images are built automatically upon changes to respective branches in GitHub.

### 3. Data Management and Training with Azure ML
- **Data Storage:**
  - Store data assets in Azure ML's data stores.

- **Pipeline Training:**
  - Define training pipelines in Azure ML.
  - Schedule training to run daily at 12 PM.
  - Automatically trigger retraining with new data.

### 4. Model Deployment
- **Using FastAPI:**
  - Develop and deploy the application using FastAPI for serving the model within a Docker container.

- **Using Azure ML Endpoints:**
  - Alternatively, deploy the model using Azure ML Endpoints for scalable and managed deployment.

## Workflow Steps

1. **Development:**
   - Implement new features or updates in the development branch.
   - Docker images for CLI, frontend, and backend are automatically built upon changes.
   
2. **Testing:**
   - Merge the development branch into the testing branch.
   - Automated unit tests run, and a code coverage report is generated.
   - Sphinx documentation is built and published on GitHub Pages.
   
3. **Deployment:**
   - Upon passing unit tests, merge the testing branch into the main branch.
   - The deployment pipeline is triggered automatically:
     - **FastAPI Deployment:** Deploy the model using FastAPI within a Docker container.
     - **Azure ML Endpoints Deployment:** Alternatively, deploy using Azure ML Endpoints.

4. **Data Management and Training:**
   - Data assets are stored in Azure ML.
   - Training pipelines are defined and scheduled to run daily at 12 PM.
   - New data triggers automatic retraining of the model.

## Visual Flow Chart

```mermaid
graph LR
    subgraph GitHub
        A[Development Branch]
        B[Testing Branch]
        C[Main Branch]
    end

    subgraph Docker
        D[CLI Docker Image]
        E[Frontend Docker Image]
        F[Backend Docker Image]
    end

    subgraph AzureML
        G[Data Storage]
        H[Training Pipeline]
        I[Managed Endpoint]
        J[Azure Monitor]
    end

    A --> B
    B --> |Unit Testing| C
    B --> |Code Coverage| G
    B --> |Sphinx Documentation| C
    C --> D
    C --> E
    C --> F
    D --> I
    E --> I
    F --> I

    H --> |Daily at 12 PM| G
    G --> |Retraining Trigger| I

    I --> J
    J --> |Alerts & Monitoring| C
