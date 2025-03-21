# Data-Driven Marketing: Predicting Term Deposit Sign-Ups for Banks

## Data Science Institute - Cohort 5 - Team 05 Project Report

In our team project, we selected the "Bank Marketing" dataset to utilize our statistical and technical skills acquired across various modules. We embarked on this project with Exploratory Data Analysis (EDA) to evaluate the correlation matrix among different variables. On the technical front, we began by identifying and eliminating outliers, performing one-hot encoding on categorical variables, and converting day/month features into cyclical variables.

## Members

- Sebastien Lozano-Forero
- Darshan Kale

## Business Case

A Portuguese bank executed a marketing campaign involving phone calls, aimed at selling term deposit subscriptions to clients. The initial phase of the campaign targeted around 45,000 clients, achieving approximately 5,000 subscriptions (~11% success rate) over a two-month period. Below is the breakdown of the estimated costs for this campaign.

##### Team Costs (Salaries)

| Role                         | Number of People | Monthly Salary | Duration (months) | Total Cost  |
|------------------------------|------------------|----------------|-------------------|-------------|
| Telemarketing Manager        | 1                | €5,000         | 2                 | €10,000     |
| Telemarketing Agents         | 18               | €3,000         | 2                 | €108,000    |
| QA Supervisor                | 1                | €4,000         | 2                 | €8,000      |
| **Subtotal (Salaries)**      |                  |                |                   | **€126,000**|

##### Operational Costs

- **Phone System & Telecommunication Charges:** €2,000

##### Total Estimated Campaign Budget

- **Total Cost:** €126,000 (Salaries) + €2,000 (Operational Costs) = **€128,000**

##### Summary

- **Team Composition:**  
  - 1 Telemarketing Manager  
  - 18 Telemarketing Agents  
  - 1 QA Supervisor
- **Duration:** 2 months  
- **Expected Call Volume:** Each agent makes approximately 60 calls per day, totaling around 2,640 calls per agent over the campaign period, thereby ensuring coverage of 45,000 clients.



### Project Objective

A subsequent iteration of this marketing campaign is planned. Leveraging the data from the first iteration, this project aims to enhance efficiencies and reduce the campaign duration while maintaining a similar success rate, thus lowering overall costs. The strategy involves developing a ranking system to prioritize potential clients based on their likelihood of subscribing to the term deposit product. By targeting the top 20% of ranked clients, we aim to achieve 90% of the expected outcomes of the entire campaign in a more cost-effective manner.

# Project Overview
- Requirements
- Exploratory Data Analysis
- Understanding the Raw Data
- Data Cleaning and Handling Missing Values

## Requirements
This project uses the following Python libraries

- pandas : For analysing and getting insights from datasets.
- seaborn : For enhancing the style of matplotlib plots.
- matplotlib : For creating graphs and plots.
- NumPy : For fast matrix operations.
- sklearn : For linear regression analysis.
- ydata_profiling : For EDA

## Exploratory Data Analysis (EDA)
Our comprehensive EDA using panda functions provided rich insights into the "Bank Marketing" dataset. By performing statistical summaries, correlation analyses, and various visualizations, we were able to uncover important relationships, trends, and potential data quality issues. The results obtained from these analyses will inform further modeling and optimization efforts, ultimately contributing to a more efficient and successful marketing campaign.

For a detailed exploration and interactive visualizations, download the EDA.ipynb file from our repository.

## Understanding the Raw Data

#### Schema

| name       | type            | description                                                                                 |
|------------|-----------------|---------------------------------------------------------------------------------------------|
| age        | number (int64)  | Age of the client                                                                           |
| balance    | number (float64)| Account balance of the client                                                               |
| campaign   | number (int64)  | Number of contacts performed during this campaign and for this client                       |
| contact    | string (object) | Contact communication type                                                                  |
| day        | number (int64)  | Last contact day of the month                                                               |
| default    | string (object) | Whether the client has credit in default                                                    |
| duration   | number (int64)  | Last contact duration, in seconds                                                           |
| education  | string (object) | Level of education of the client                                                            |
| housing    | string (object) | Whether the client has a housing loan                                                       |
| job        | string (object) | Job type of the client                                                                      |
| loan       | string (object) | Whether the client has a personal loan                                                      |
| marital    | string (object) | Marital status of the client                                                                |
| month      | string (object) | Last contact month of the year                                                              |
| pdays      | number (int64)  | Number of days since the client was last contacted from a previous campaign                 |
| poutcome   | string (object) | Outcome of the previous marketing campaign                                                  |
| previous   | number (int64)  | Number of contacts performed before this campaign and for this client                       |
| y          | string (object) | Whether the client has subscribed to a term deposit                                         |

### Understanding the Features

| column name | feature                         | description                                                                                                        |
|-------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------|
| age         | Age                             | The age of the client, which may impact financial needs and decisions.                                             |
| balance     | Account Balance                 | The balance amount in the client's account. Generally indicates financial stability.                               |
| campaign    | Campaign Contacts               | The number of contacts made during the current marketing campaign for the client.                                  |
| contact     | Contact Communication Type      | The type of communication used to reach the client (e.g., cellular, telephone).                                    |
| day         | Last Contact Day                | The day of the month the last contact was made.                                                                    |
| default     | Credit Default History          | Indicates if the client has defaulted on credit before.                                                            |
| duration    | Contact Duration                | Duration in seconds of the last contact. Higher duration usually indicates better engagement.                      |
| education   | Educational Level               | Level of education attained by the client.                                                                         |
| housing     | Housing Loan Status             | Whether the client has a housing loan. May indicate long-term financial commitments.                               |
| job         | Job Type                        | The type of job the client has.                                                                                     |
| loan        | Personal Loan Status            | Whether the client has a personal loan.                                                                            |
| marital     | Marital Status                  | The marital status of the client.                                                                                  |
| month       | Last Contact Month              | The month in which the last contact was made. Seasonal effects might be observed.                                  |
| pdays       | Days Since Prior Contact        | The number of days that passed since the client was last contacted in a previous campaign. -1 means not contacted before.|
| poutcome    | Previous Campaign Outcome       | The outcome of the previous campaign contacts (e.g., success, failure).                                             |
| previous    | Previous Contacts               | The number of contacts before the current campaign. Helps in understanding engagement history.                     |
| y           | Term Deposit Subscription       | Whether the client has subscribed to a term deposit (Yes/No). This is the target variable.                         |

### Summarizations of the Dataset

The following table summarizes the key aspects derived from the Bank Marketing dataset, providing foundational insights including the extent of client information covered, the temporal dimension, and the completion status of the data.

| question                                | analysis                                                                                       |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
| How many clients are in this data set?  | The dataset consists of approximately 45,000 client records.                                   |
| Over what period is this data collected?| The data is collected over a span of several years, with specific contact months noted.        |
| What is the range of transaction values?| The account balance ranges from negative values to large positive amounts.                     |
| What is the total number of observations?| There are about 45,000 observations in the dataset.                                            |
| Are there any missing values?           | There are minimal missing values, and appropriate handling techniques such as imputation or exclusion may be needed. |

These tables provide a structured view of the dataset, detailing the schema of the raw data, an understanding of each feature's role, and key summarizations that highlight the data's scope and completeness.

## Data Cleaning and Handling Missing Values

![image](https://github.com/user-attachments/assets/3c1b0b1f-5c38-4fe6-813d-eb42c79c613a)


In preparation for analysis, the dataset underwent several key data cleaning steps to handle missing values, encode categorical variables, and mitigate the influence of outliers.

1. **Trimming Extreme Values (Outliers)**:
   - To reduce the impact of extreme outliers, we filtered the `balance` and `duration` columns. Records with `balance` less than 10,000 and `duration` less than 1,800 seconds were retained. This helps in maintaining a clean dataset by excluding potentially anomalous data points.

2. **Handling Missing or Specific Values in `pdays`**:
   - For the `pdays` column (number of days since the client was last contacted in a previous campaign), entries with a value of `-1` were replaced with `0`, indicating no prior contact; all others were marked as `1`.

3. **Binary Encoding of Categorical Variables**:
   - The dataset contains several binary categorical variables (`default`, `housing`, `loan`, and `y`). These were encoded to numerical values where `yes` was encoded as `1` and `no` was encoded as `0`.

4. **Converting Month Names to Numerical Values**:
   - The `month` column (indicating the last contact month) originally contained month names. These were converted to their respective numerical values for easier temporal analysis.

5. **Creating Cyclical Features for `day` and `month`**:
   - To capture the cyclical nature of time-related features, new columns representing sine and cosine transformations of `day` and `month` were introduced. This transformation helps in preserving the cyclical relationships (e.g., December and January are close).

Through these cleaning steps, the dataset was prepared for subsequent analysis, ensuring that it is free from extreme outliers, missing critical values, and encoded appropriately for modeling and visualization techniques.

## Correlation between features

![image](https://github.com/user-attachments/assets/fd93363e-5992-480b-905e-a1f5190ace08)

The correlation matrix presented provides insight into the linear relationships between various features within the dataset, with a specific focus on the target variable, `y`, which indicates term deposit subscription (0 or 1). Key observations include:

1. **Duration**: The feature `duration` shows a strong positive correlation with the target variable `y`, indicating that longer contact durations are significantly associated with a higher probability of subscription to a term deposit.
  
2. **Previous and Pdays**: There is a notable positive correlation between `previous` and `pdays`. This relationship suggests that clients who had a higher number of previous contacts also have shorter intervals since their last contact in the previous campaign, indicating ongoing engagement.

3. **Campaign and Duration**: A slight negative correlation exists between `campaign` and `duration`. This suggests that an increased number of contacts within a single campaign might be associated with shorter individual contact durations, possibly indicating less effective engagement when clients are contacted too frequently.

4. **Housing and Loan**: Both `housing` and `loan` features are weakly correlated with each other, potentially implying that clients with housing loans might also have personal loans, albeit to a small extent.

5. **Balance**: The `balance` feature has a weak positive correlation with `y`. While higher account balances slightly influence the likelihood of subscribing to a term deposit, the correlation is weak, suggesting other factors might be more crucial in the decision-making process.

6. **Age**: There is a weak negative correlation between `age` and term deposit subscription (`y`). This suggests that older clients have a slightly lower tendency to subscribe, although this impact is not substantial.

7. **Job and Education**: Employment (`job`) and educational attainment (`education`) features show very low correlations with the target variable, indicating that while these demographic factors are crucial, they do not strongly influence whether a client subscribes to a term deposit.

8. **Marital Status**: The feature `marital` shows very low to negligible correlation with the target variable `y`, implying that being single, married, or divorced does not significantly affect a client's term deposit subscription status.

In summary, `duration` emerges as the most influential feature positively correlated with term deposit subscriptions, while other features like `previous`, `balance`, and demographic attributes exhibit much weaker correlations with `y`. This matrix underscores the need for including both strong and weakly correlated features in model development to capture the complex relationships influencing term deposit subscriptions.

## Model Development
A robust pipeline was designed to streamline preprocessing and modeling. The pipeline involved:

#### 1. Preprocessing with StandardScaler and OneHotEncoder:
We used ColumnTransformer to standardize numerical columns and encode categorical columns.

#### 2. Machine Learning Models:
Multiple classifiers were trained, including Logistic Regression, Random Forest, XGBoost, LightGBM, K-Nearest Neighbors, Naive Bayes, and others.

## Handling Imbalanced Data
Given the imbalance in the target variable, the SMOTE technique (Synthetic Minority Over-sampling Technique) was applied to achieve a balanced dataset.

## Model Training and Evaluation
Overall Assessment Metrics Interpretation
The overall assessment metrics provide a comprehensive evaluation of the model's performance on the training and testing datasets. The key metrics are accuracy, Cohen's kappa, log loss, and F1 score.

#### Accuracy Score
Training Accuracy: The model achieved a near-perfect accuracy score of 0.992 (99.24%) on the training set, indicating excellent performance in predicting term deposit subscriptions for the training data.
Testing Accuracy: The testing accuracy is slightly lower at 0.950 (95.05%), suggesting that the model maintains high performance on unseen data, though with some reduction due to variability in the test set.
#### Cohen's Kappa
Training Cohen's Kappa: The Cohen's kappa score of 0.985 (98.47%) on the training set indicates very high agreement between predicted and actual labels, confirming the model's robustness in handling classification tasks.
Testing Cohen's Kappa: Similarly, the testing Cohen's kappa score is 0.901 (90.09%), demonstrating good agreement and consistency in predictions and further validating the model's effectiveness on new data.
#### Log Loss
Training Log Loss: The log loss value of 0.047 suggests the model's predictions for the training data are highly confident and well-calibrated.
Testing Log Loss: A slightly higher log loss of 0.112 on the test data indicates some uncertainty, but the values remain low, showcasing good prediction accuracy and confidence overall.
#### F1 Score
Training F1 Score: The F1 score of 0.992 (99.23%) indicates a balanced performance in terms of precision and recall on the training set, highlighting the model's ability to correctly identify positive subscriptions with minimal false positives and false negatives.
Testing F1 Score: The test F1 score of 0.950 (95.00%) reflects the model's competence in maintaining this balanced performance on unseen data, affirming its reliability and effectiveness in real-world scenarios.
#### Summary
The metrics reveal an outstanding model performance, both on the training and testing datasets:

High Accuracy: Reflects the model's precise predictions.
High Cohen's Kappa: Indicates strong agreement with the actual outcomes.
Low Log Loss: Suggests well-calibrated predictions.
High F1 Score: Demonstrates balanced precision and recall.
Despite slight reductions in the test metrics compared to training metrics (which are expected), the model retains substantial predictive power and reliability. This consistency ensures its applicability for efficiently predicting term deposit subscriptions for future marketing campaigns.
