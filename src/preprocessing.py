"""
Data Preprocessing Utilities

Author: Sakshi Raut
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

def impute_missing_values(df, columns, strategy="median"):
    """
    Impute missing values for selected columns.
    """

    imputer = SimpleImputer(strategy=strategy)

    df[columns] = imputer.fit_transform(df[columns])

    return df

def detect_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    return df[
        (df[column] < lower)
        |
        (df[column] > upper)
    ]

def cap_outliers(df, column):

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    df[column] = np.where(

        df[column] < lower,

        lower,

        np.where(

            df[column] > upper,

            upper,

            df[column]

        )

    )

    return df

def encode_categorical(df):

    encoder = LabelEncoder()

    categorical = df.select_dtypes(
        include="object"
    ).columns

    encoders = {}

    for column in categorical:

        df[column] = encoder.fit_transform(
            df[column]
        )

        encoders[column] = encoder

    return df, encoders

def scale_numeric(df, columns):

    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(
        df[columns]
    )

    return df, scaler

def split_features_target(df, target):

    X = df.drop(columns=[target])

    y = df[target]

    return X, y

def split_train_test(
    X,
    y,
    test_size=0.20,
    random_state=42
):

    return train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )

def save_processed_dataset(df, filepath):

    df.to_csv(

        filepath,

        index=False

    )

    print("Dataset saved successfully.")