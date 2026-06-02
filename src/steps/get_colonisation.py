import numpy as np 
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
INTERIM_DIR = DATA_DIR / 'interim'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import data_cleaning_tools as dct

cfg = dct.load_config(PROJECT_ROOT /'configs'/'config.yaml')

def get_colonisation(
    main_df,
    cfg = cfg,
    join_cols=("subject", "admission_date"),
    new_col_name="colonisation",
):
    """
    Add a binary colonisation feature to the main dataframe.

    A patient/admission is flagged as colonised if there is at least one
    matching screening record for that subject and admission date.

    If no matching screening record is found, colonisation is assumed to be 0.

    Parameters
    ----------
    main_df : pd.DataFrame
        Main analysis dataframe.

    cfg_paths : dict
        Dictionary containing the path to the screens CSV under cfg_paths["screens"].

    join_cols : tuple or list of str, default ("subject", "admission_date")
        Columns used to match screening records to the main dataframe.

    new_col_name : str, default "colonisation"
        Name of the binary colonisation feature.

    Returns
    -------
    pd.DataFrame
        Copy of main_df with a binary colonisation column.
    """

    main_df = main_df.copy()

    if isinstance(join_cols, str):
        join_cols = [join_cols]
    else:
        join_cols = list(join_cols)

    cfg_paths = cfg['paths']

    screens_df = pd.read_csv(cfg_paths["screens"])
    screens_df = dct.clean_df_columns(screens_df)

    screening_sites = ["rectal swab", "rectum", "rec", "recs"]

    ekp_organisms = [
        "escherichia coli",
        "klebsiella pneumoniae",
        "klebsiella oxytoca",
        "proteus mirabilis",
    ]

    screens_df["site"] = screens_df["site"].str.lower().str.strip()
    screens_df["organism_bug"] = screens_df["organism_bug"].str.lower().str.strip()

    screens_df = screens_df[
        screens_df["site"].isin(screening_sites)
        & screens_df["organism_bug"].isin(ekp_organisms)
    ].copy()

    exposed = (
        screens_df[join_cols]
        .drop_duplicates()
        .assign(**{new_col_name: 1})
    )

    main_df['admission_date'] = pd.to_datetime(main_df['admission_date'])
    

    main_df = main_df.merge(
        exposed,
        on=join_cols,
        how="left"
    )

    main_df[new_col_name] = main_df[new_col_name].fillna(0).astype(int)

    return main_df