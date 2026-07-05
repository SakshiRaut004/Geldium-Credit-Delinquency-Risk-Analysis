# 📊 Geldium Credit Delinquency Risk Analysis

> **End-to-End Credit Risk Analytics Project**
>
> Exploratory Data Analysis • Machine Learning • Power BI • Python

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811)
![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project demonstrates an end-to-end credit risk analytics workflow using Geldium's customer delinquency dataset.

The objective is to analyze customer financial behavior, identify potential risk indicators, and prepare the dataset for predictive modeling through a structured analytics pipeline.

The project follows industry-standard analytics practices, including:

- Business Understanding
- Exploratory Data Analysis (EDA)
- Data Quality Assessment
- Data Preprocessing
- Feature Engineering
- Machine Learning
- Model Evaluation
- Business Intelligence Dashboarding

## 🚀 Project Progress

| Phase | Status |
|---------|--------|
| Repository Setup | ✅ Completed |
| Exploratory Data Analysis | ✅ Completed |
| Business Report | ✅ Completed |
| Data Preprocessing | 🚧 In Progress |
| Feature Engineering | ⏳ Planned |
| Machine Learning | ⏳ Planned |
| Model Evaluation | ⏳ Planned |
| Power BI Dashboard | ⏳ Planned |
| Final Documentation | ⏳ Planned |

## 💼 Business Problem

Financial institutions process thousands of loan and credit applications every day. One of the biggest challenges is identifying customers who are at risk of becoming delinquent before missed payments occur.

Traditional manual assessment methods are often time-consuming and may overlook subtle behavioral patterns hidden within customer data.

This project explores customer demographic, financial, and repayment information to identify potential risk indicators that contribute to credit delinquency. The insights generated through exploratory data analysis and predictive modeling can help financial institutions improve risk assessment, reduce loan defaults, and support proactive intervention strategies.

## 🎯 Project Objectives

### Business Objectives

- Understand customer financial behavior.
- Identify early indicators of credit delinquency.
- Support risk-based decision-making.
- Improve future delinquency prediction models.

### Technical Objectives

- Perform comprehensive exploratory data analysis.
- Assess data quality and handle missing values.
- Engineer meaningful predictive features.
- Build and evaluate multiple machine learning models.
- Develop an interactive Power BI dashboard.

## 📂 Dataset Overview

| Attribute | Details |
|------------|---------|
| Domain | Banking & Financial Services |
| Problem Type | Binary Classification |
| Records | 500 |
| Features | 19 |
| Target Variable | Delinquent_Account |
| Missing Values | Income, Loan Balance, Credit Score |
| Source | Tata iQ Virtual Experience (Simulation Dataset) |

## 📁 Repository Structure

```text
Geldium-Credit-Delinquency-Risk-Analysis
│
├── assets/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
├── src/
├── visuals/
├── models/
├── prompts/
├── README.md
└── requirements.txt
```

## 🔄 Analytics Workflow

```mermaid
flowchart LR

A[Business Understanding]

-->B[Data Understanding]

-->C[EDA]

-->D[Data Cleaning]

-->E[Feature Engineering]

-->F[Machine Learning]

-->G[Model Evaluation]

-->H[Power BI Dashboard]

-->I[Business Recommendations]
```

## 📊 Exploratory Data Analysis

The exploratory analysis focused on:

- Dataset profiling
- Missing value assessment
- Distribution analysis
- Outlier detection
- Categorical analysis
- Bivariate analysis
- Correlation analysis
- Risk factor identification

**Key Outputs**

- Target Distribution
- Correlation Heatmap
- Credit Utilization Analysis
- Customer Demographics
- Employment Analysis

## 💡 Key Findings

- Only three variables contained missing values, all of which can be addressed using median imputation.
- No duplicate records were identified.
- The dataset exhibits moderate class imbalance with approximately 16% delinquent customers.
- Individual numerical variables display weak linear relationships with delinquency, indicating that predictive performance may improve through feature engineering and non-linear machine learning models.
- Credit utilization, payment behavior, and debt-related metrics remain valuable candidate predictors for future modeling.