# Training Cost Expectations

## 1. Data Preparation Costs
- **Data Acquisition:**
  - We obtained all necessary data from NPEC company at no cost, as part of our collaborative study project.
- **Data Labeling:**
  - Data science and artificial intelligence students, along with our team, annotated the datasets for free. Additionally, we improved our data by labeling some additional images as a team.
- **Data Storage:**
  - We estimate needing around 200 GB of storage. However, if we require more, it won't pose a problem, as we'll leverage Azure's pay-as-you-go data storage pricing model.
    - Storage: 200 GB
    - Cost: €0,017 per GB/month
    - Monthly Cost: €3,40

## 2. Azure ML Resource Costs
We will utilize the Standard NC6 instance for training our models. This instance provides 6 vCPUs, 56 GB of RAM, and a single K80 GPU.

### Compute Instance Options:

| Instance   | vCPU(s) | RAM    | GPU    | Pay As You Go | 1 Year Savings Plan | 3 Year Savings Plan |
|------------|---------|--------|--------|---------------|---------------------|---------------------|
| NC6        | 6       | 56 GB  | 1X K80 | €1,079/hour   | €0,77/hour (~29% savings) | €0,57/hour (~48% savings) |
| NC12       | 12      | 112 GB | 2X K80 | €2,158/hour   | €1,53/hour (~29% savings) | €1,13/hour (~48% savings) |
| NC24       | 24      | 224 GB | 4X K80 | €4,315/hour   | €3,05/hour (~29% savings) | €2,25/hour (~48% savings) |
| NC6s v3    | 6       | 112 GB | 1X V100| €3,536/hour   | €2,60/hour (~27% savings) | €1,89/hour (~47% savings) |
| NC12s v3   | 12      | 224 GB | 2X V100| €7,071/hour   | €5,19/hour (~27% savings) | €3,77/hour (~47% savings) |
| NC24s v3   | 24      | 448 GB | 4X V100| €14,141/hour  | €10,37/hour (~27% savings) | €7,53/hour (~47% savings) |
| NC24rs v3  | 24      | 448 GB | 4X V100| €15,555/hour  | €11,41/hour (~27% savings) | €8,28/hour (~47% savings) |

**Chosen Instance: NC6**

- **Cost:**
  - Pay-as-you-go pricing at €1,079 per hour.
  - Training Time: We estimate an initial model training time of 10 hours.
  - Cost: 10 * €1,079 = €10,79

- **Storage Costs:**
  - Description: We'll also require additional storage for storing intermediate data and models during the training process.
  - Additional Storage: Approximately 100 GB.
  - Monthly Cost: 100 * €0,017 = €1,70

## 3. Model Training Costs
- **Training the ResNet Model:**
  - Description: The model training involves training the ResNet model based on the chosen architecture.
  - Initial Training: We anticipate the initial training to take 10 hours on the NC6 instance.
  - Cost: As previously mentioned, the cost for 10 hours of training on the NC6 instance is estimated at €10,79.
  - Specific Training Durations:
    - ResNet-50: 4 hours * €1,079/hour = €4,32
    - ResNet-101: 6 hours * €1,079/hour = €6,47
  - Total Training for 10 sessions: 5 sessions of ResNet-50 and 5 sessions of ResNet-101
  - Total Cost: (5 * €4,32) + (5 * €6,47) = €21,60 + €32,35 = €53,95

## 4. Model Deployment Cost
- **Deploying the Model:**
  - Description: We'll deploy the trained model using Azure Kubernetes Service (AKS) to serve predictions.
  - Cost: €0,093 per cluster per hour.
  - Uptime: 24/7.
  - Monthly Cost: €0,093 * 24 * 30 = €67,00.
- **Inference Costs:**
  - Description: Cost associated with serving predictions.
  - Example: 10.000 predictions per month.
  - Cost per prediction: €0,0001 per prediction.
  - Monthly Cost: 10.000 * €0,0001 = €1,00

## 5. User Training Costs
- **User Training on the Application:**
  - Description: Users will have the capability to train their own models within the application.
  - Additional Compute: We assume a similar usage pattern to the initial model training.
    - Cost per User: 10 hours per user * €1,079 (NC6 instance).
    - Estimated Monthly Users: 3 users.
  - Monthly Cost Calculation:
    - Monthly Cost = (Number of users) * (Number of hours per user) * (Cost per hour per user).
    - Monthly Cost = 3 * 10 * €1,079.
    - Monthly Cost = €32,37.

## 6. Monitoring and Maintenance Costs
- **Monitoring Performance:**
  - Monthly Cost: This will depend on the volume of data ingested for monitoring purposes.
    - For Basic Logs (Pay-As-You-Go): €0,602 per GB.
    - For Analytic Logs (Pay-As-You-Go): €2,765 per GB.
  - Monthly Cost for Basic Logs: 5 GB * €0,602 per GB = €3,01.
  - Monthly Cost for Analytic Logs: 5 GB * €2,765 per GB = €13,83.
- **Retraining Models:**
  - Retraining Frequency: Every 3 months.
  - Cost: This will be the cost of additional compute instances for retraining models.
    - We previously estimated the cost for initial model training on the NC6 instance at €1,079 per hour.
    - Assuming a similar duration of 10 hours for retraining, the cost per retraining session would be €10,79.
  - Monthly Amortized Cost: Since retraining occurs every 3 months, we'll divide the cost by 3 to get the monthly amortized cost.
    - Monthly Amortized Cost = €10,79 / 3 = €3,60.

So, the estimated monitoring and maintenance costs per month are as follows:
- Monitoring Performance: €3,01 (Basic Logs) + €13,83 (Analytic Logs) = €16,84.
- Retraining Models: €3,60.
Therefore, the total estimated monthly monitoring and maintenance costs are approximately €20,44.

## Total Estimated Monthly Costs:
1. Data Storage: €3,40
2. Azure ML Compute: €10,79
3. Additional Storage: €1,70
4. Model Deployment: €67,00
5. Inference: €1,00
6. User Training: €32,37
7. Monitoring and Maintenance: €20,44

Total Monthly Cost: €3,40 + €10,79 + €1,70 + €67,00 + €1,00 + €32,37 + €20,44 = €137,00

## Summary:
- Initial One-Time Costs: None
- Monthly Recurring Costs: €137,00

## References:
- [Azure Machine Learning Pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/)
- [Azure Pricing Details](https://azure.microsoft.com/en-us/pricing/details)
