import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Dict
import re



"""
Collection of helper functions for data preprocessing, 
exploratory analysis, and visualization in the ESBL project.
"""


# ========================================
# EDA Functions
# ========================================


def count_by_esbl_status(df:pd.DataFrame, col:str, keys):

    """
    Splits a column of a dataframe into ESBL and non-ESBL subgroups
    (based on the 'esbl_status' column), counts values, and calculates percentages.

    # percentages are showing the distribution for that column in the ESBL or non-ESBL group

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain a column named 'esbl_status'
        with values 'ESBL' or 'non-ESBL'.
    col : str
        The column whose value counts you want to split and compare.

    keys : unique key identifiers

    Returns
    -------
    pd.DataFrame
        A dataframe with:
            - `index` : unique values of `col`
            - `ESBL count`, `ESBL pct` : counts and percentages for ESBL rows
            - `non-ESBL count`, `non-ESBL pct` : counts and percentages for non-ESBL rows
    """
    
    counts = (
    df.drop_duplicates(keys)["esbl_status"]
    .value_counts()
    )

    esbl_total = counts.get("ESBL", 0)
    non_esbl_total = counts.get("non-ESBL", 0)


    esbl_split = (
                df[df['esbl_status'] == 'ESBL'][col]
                .value_counts()
                .rename_axis(col)
                .reset_index(name = 'ESBL_count')
    )

    non_esbl_split = (
                    df[df['esbl_status'] == 'non-ESBL'][col]
                    .value_counts()
                    .rename_axis(col)
                    .reset_index(name = 'non-ESBL_count')
    )
    
    esbl_split['ESBL_pct'] = round((esbl_split['ESBL_count']*100)/esbl_total, 1)
    non_esbl_split['non-ESBL_pct'] = round((non_esbl_split['non-ESBL_count']*100)/non_esbl_total, 1)

    total_counts = pd.merge(esbl_split, non_esbl_split, on=col, how="outer").fillna(0)

    return total_counts  


# ========================================
# Plotting Functions
# ========================================



def plot_esbl_distribution(df: pd.DataFrame, col: str, pct: bool):
    """
    Plots bar charts showing the distribution of a specified column across ESBL and non-ESBL cohorts.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing ESBL-related data.
    col : str
        The column name to analyze and plot.
    pct : bool
        If True, plots proportions for ESBL and non-ESBL cohorts.
        If False, plots raw counts for ESBL, non-ESBL, and total distribution.

    Raises:
    -------
    ValueError
        If required columns are missing or if filtered data is empty.

    Returns:
    -------
    None
        Displays the plots using matplotlib.
    """

# The function assumes 'df' is the 'total_counts' DataFrame returned by the previous function.

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    # Check for required percentage columns (pct is assumed True for this function)
    required_pct_cols = ['ESBL_pct', 'non-ESBL_pct']

    for c in required_pct_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' is missing for percentage plots.")

    if df[required_pct_cols].dropna().empty:
        raise ValueError("No data available for percentage plotting.")

# NOTE: To use this, you would call it like:
# plot_esbl_comparison(total_counts, 'medication_name_short')

    else:

        fig, axes = plt.subplots(1,3, figsize = (16,8))

        if pct:

            df['ESBL_pct'].plot(kind='bar', ax=axes[0], color='lightcoral')
            axes[0].set_title(f'{col} distribution (ESBL cohort)')
            axes[0].set_xlabel(col)
            axes[0].tick_params(axis='x', rotation=90)

        # non-ESBL cohort
            df['non-ESBL_pct'].plot(kind='bar', ax=axes[1], color='skyblue')
            axes[1].set_title(f'{col} distribution (non-ESBL cohort)')
            axes[1].set_xlabel(col)
            axes[1].tick_params(axis='x', rotation=90)

            df['total_pct'] = df['ESBL_pct'] + df['non-ESBL_pct']
            df['total_pct'].plot(kind='bar', ax=axes[2],
            color='gray', legend=False).plot(kind='bar', ax=axes[2], color='gray')
            axes[2].set_title(f'Total {col} distribution')
            axes[2].set_xlabel(col)
            axes[2].tick_params(axis='x', rotation=90)


    plt.tight_layout()
    plt.show()


