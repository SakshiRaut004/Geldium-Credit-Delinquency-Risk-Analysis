"""
Feature Engineering Utilities

Author: Sakshi Raut
Project: Geldium Credit Delinquency Risk Analysis

Description:
Reusable feature engineering functions for transforming cleaned
credit risk data into model-ready features.
"""

import pandas as pd
import numpy as np

def create_income_group(df):

    bins = [
        0,
        50000,
        100000,
        150000,
        np.inf
    ]

    labels = [
        "Low",
        "Middle",
        "Upper Middle",
        "High"
    ]

    df["Income_Group"] = pd.cut(
        df["Income"],
        bins=bins,
        labels=labels
    )

    return df

def create_credit_score_band(df):

    bins = [
        300,
        580,
        670,
        740,
        800,
        850
    ]

    labels = [
        "Poor",
        "Fair",
        "Good",
        "Very Good",
        "Excellent"
    ]

    df["Credit_Score_Band"] = pd.cut(
        df["Credit_Score"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

def create_utilization_level(df):

    bins = [
        0,
        0.30,
        0.50,
        0.75,
        np.inf
    ]

    labels = [
        "Low",
        "Moderate",
        "High",
        "Critical"
    ]

    df["Utilization_Level"] = pd.cut(
        df["Credit_Utilization"],
        bins=bins,
        labels=labels
    )

    return df

def create_debt_burden(df):

    bins = [
        0,
        0.20,
        0.35,
        0.50,
        np.inf
    ]

    labels = [
        "Low",
        "Moderate",
        "High",
        "Severe"
    ]

    df["Debt_Burden"] = pd.cut(
        df["Debt_to_Income_Ratio"],
        bins=bins,
        labels=labels
    )

    return df

def create_age_group(df):

    bins = [
        18,
        30,
        45,
        60,
        np.inf
    ]

    labels = [
        "Young Adult",
        "Adult",
        "Middle Age",
        "Senior"
    ]

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

def create_loan_to_income_ratio(df):

    df["Loan_to_Income_Ratio"] = (
        df["Loan_Balance"] /
        df["Income"]
    )

    return df

def create_missed_payment_rate(df):

    df["Missed_Payment_Rate"] = (
        df["Missed_Payments"] / 6
    )

    return df

def create_financial_stress_index(df):

    utilization = df["Credit_Utilization"]

    dti = df["Debt_to_Income_Ratio"]

    missed = df["Missed_Payment_Rate"]

    credit = 1 - (
        (df["Credit_Score"] - 300) / (850 - 300)
    )

    df["Financial_Stress_Index"] = (
        utilization +
        dti +
        missed +
        credit
    ) / 4

    return df

def create_risk_score(df):

    score = np.zeros(len(df))

    score += (df["Credit_Utilization"] > 0.80).astype(int)

    score += (df["Debt_to_Income_Ratio"] > 0.40).astype(int)

    score += (df["Missed_Payments"] >= 4).astype(int)

    score += (df["Credit_Score"] < 500).astype(int)

    df["Risk_Score"] = score

    return df

def create_risk_category(df):

    conditions = [
        df["Risk_Score"] <= 1,
        df["Risk_Score"] == 2,
        df["Risk_Score"] >= 3
    ]

    categories = [
        "Low",
        "Medium",
        "High"
    ]

    df["Risk_Category"] = np.select(
        conditions,
        categories,
        default="Low"
    )

    return df
