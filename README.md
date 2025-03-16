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
