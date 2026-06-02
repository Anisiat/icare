import yaml, os
import pandas as pd
import numpy as np 
import re
import logging 
import sys
from pathlib import Path
# -----------------------------
# Logging config (to stdout)
# -----------------------------

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")


def remove_unknowns(df):
    df = df.copy()
    
    unknown_values = [
        "other",
        "other_unknown",
        "unknown",
        "unk",
        "not known",
        "not specified",
        "missing",
        "none",
        "",
    ]

    df = df.replace(
        to_replace=unknown_values,
        value=pd.NA
    )
    return df

# -----------------------------
# Config (no pathlib)
# -----------------------------


def load_config(cfg_path="configs/config.yaml"):
    """
    Load YAML file.

    If a 'paths' section exists, convert relative paths
    to absolute paths relative to the YAML file location.
    """

    cfg_path = Path(cfg_path).resolve()

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve paths if present
    if "paths" in cfg:

        yaml_dir = cfg_path.parent

        cfg["paths"] = {
            key: (
                Path(value)
                if Path(value).is_absolute()
                else yaml_dir / value
            )
            for key, value in cfg["paths"].items()
        }

    return cfg


# -----------------------------
# Helpers & Cleaning
# -----------------------------

def clean_df_columns(
    df:pd.DataFrame,
    drop_cols = ('cefixime', 'ceftolozane-tazobactam'),
    datetime_keywords = ('dt', 'date', 'time')):

    """
    Normalize column names, drop unwanted columns, and parse datetime-like columns.
    """
    
    df_out = df.copy()

    def _normalise(name:str) -> str:
        name = name.replace("'", "").replace('"', '').strip().lower()
        # name = re.sub(r"\s+", "_", name) #this means replace spaces (\s is space and then + means one or more)
        # name = re.sub(r"[^0-9a-zA_Z]", "_", name) # replace any character that is no 0-9, a-z or A-Z
        # name = re.sub(r"__+", "_", name).strip("_")
        return name 

    old_cols = df_out.columns
    new_cols = [_normalise(c) for c in df_out.columns]
    df_out.columns = new_cols 
    renamed_cols = dict(zip(old_cols, new_cols))
    
    logging.info(f"Renamed columns: {renamed_cols}")

    # drop useless columns 
    to_drop = [c for c in df_out.columns if c in set(drop_cols)]

    if to_drop:
        df_out = df_out.drop(to_drop, axis = 1)
        logging.info(f"dropped columns {to_drop}")

    # convert date columns to datetime

    for col in df_out.columns:
        if any(k.lower() in col for k in datetime_keywords):
            try:
                df_out[col] =  pd.to_datetime(df_out[col], errors = 'coerce')
                logging.info(f"Converted {col} to datetime")
            except Exception as e: 
                logging.warning(f"Could not convert {col}: {e}")
    
    
    logging.info("Column cleaning finished")
    return df_out



def map_to_dict(map_df, value_col, key_col = 'code'):

    map_df[key_col] = map_df[key_col].str.lower().str.strip()
    map_df[value_col] = map_df[value_col].str.lower().str.strip()
    map_dict = map_df.set_index(key_col)[value_col].to_dict()
    
    return map_dict


def map_codes(df, mapping_dict, col, new_col):

    df[col] = df[col].astype('string').str.strip()
    df[new_col] = df[col].map(mapping_dict)

    # check if all codes are mapped 
    map_codes = list(mapping_dict.keys())
    df_codes = df[col].unique()

    print(f"Code(s) not mapped: {[code for code in df_codes if code not in map_codes]}")

    return df

    
# ========================================
# Data Cleaning Functions
# ========================================

def clean_col_names(df):

    # get rid of the '' around column names 
    df.columns = [col.replace("'", "") for col in df.columns]

    for col in df.columns:
        if any(sub in col.lower() for sub in ['date', 'dt', 'time']):
            df[col] = pd.to_datetime(df[col], errors = 'coerce')
            success_rate = df[col].notnull().mean()
            print(f"Converted {col} to datetime : {success_rate:.2%} datetime") #check how many were converted successfully 

    df.columns = df.columns.str.lower()

    return df


def map_col_values(df, mapping_source, col, drop_unmapped=False):
    """
    Standardises the values of a given column by mapping variants/synonyms to canonical names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the column to standardise.
    mapping_dict : dict
        Dictionary where:
            - keys are the canonical names
            - values are lists of variant strings (synonyms) that should be mapped to the canonical name.
    col : str
        The column in `df` to standardise.

    Returns
    -------
    df : pd.DataFrame
        A copy of the dataframe with the column values standardised.
    unmapped : list
        A list of unique values in the column that were not mapped to any canonical name.
    """


    mapping_dict = mapping_source
        
    df = df.copy()  # avoid modifying the original dataframe
    col_mapping = {}

    # Build the mapping dictionary dynamically using regex matches
    for canonical, variants in mapping_dict.items():
        for variant in variants:
            regex = re.compile(rf"\b{variant}\b", flags=re.IGNORECASE)
            matches = [x for x in df[col].dropna().unique() if regex.search(str(x))]
            for match in matches:
                col_mapping[match] = canonical

    # Apply mapping
    df[col] = df[col].map(col_mapping).fillna(df[col])

    # Identify values that weren’t mapped
    unmapped = [val for val in df[col].dropna().unique() if val not in mapping_dict]

    if drop_unmapped:
        df = df[df[col].isin(mapping_dict.keys())].copy()
        unmapped = []  # no need to return unmapped if we dropped everything


    return df