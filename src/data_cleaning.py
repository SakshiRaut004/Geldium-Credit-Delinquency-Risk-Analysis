"""
Data Cleaning Utilities
"""

import pandas as pd
import numpy as np

def load_dataset(filepath):
    """
    Load dataset from CSV.
    """

    return pd.read_csv(filepath)

def dataset_summary(df):

    print("="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nData Types\n")

    print(df.dtypes)

    return None

def missing_value_summary(df):

    missing = pd.DataFrame({

        "Missing Values": df.isnull().sum(),

        "Percentage":
        round((df.isnull().sum()/len(df))*100,2)

    })

    return (
        missing
        .query("`Missing Values` > 0")
        .sort_values(
            by="Percentage",
            ascending=False
        )
    )

def duplicate_summary(df):

    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicates}")

    return duplicates

def numerical_summary(df):

    return df.describe().T

def categorical_summary(df):

    categorical = df.select_dtypes(include="object").columns

    for column in categorical:

        print("="*60)

        print(column)

        print("="*60)

        print(df[column].value_counts())

        print()

def datatype_summary(df):

    return pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.values

    })

def memory_usage(df):

    memory = df.memory_usage(deep=True).sum()/1024**2

    print(f"Dataset Memory Usage : {memory:.2f} MB")

    return memory

def unique_values(df):

    return pd.DataFrame({

        "Unique Values":

        df.nunique()

    }).sort_values(

        by="Unique Values"

    )

import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(df):

    plt.figure(figsize=(12,5))

    sns.heatmap(

        df.isnull(),

        cbar=False,

        cmap="viridis"

    )

    plt.title("Missing Value Heatmap")

    plt.show()