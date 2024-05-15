import pandas as pd
import numpy as np

def summarize_categorical(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_summary = {}
    for col in categorical_cols:
        categorical_summary[col] = df[col].value_counts()
    return categorical_summary

def describe_numerical(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_stats = df[numerical_cols].describe()
    return numerical_stats