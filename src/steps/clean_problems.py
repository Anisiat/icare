import numpy as np 
import pandas as pd
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
INTERIM_DIR = DATA_DIR / 'interim'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import data_cleaning_tools as dct
import yaml


# def clean_problems(df, problems_yaml):

#     df = df.copy()

#     df = clean_col_names(df)

#     df.drop('encntr_id', axis=1, inplace =True)
    
#     try:
#         df = df.dropna(subset = 'problem_dt_tm', axis = 0)
#     except KeyError():
#         logging.error('problem_dt_tm is not a column in the problems_df')
#         return 

#     df = df[~df['problem_desc'].isin(problems_yaml['not_comorbidities'])]

#     # some people may have more than one comorbidity so need to pivot problems table before merging
    
#     # 1. Keep only what we need from problems_df
#     problems_min = df[['subject', 'problem_desc']].dropna()
    
#     # (Optional) if you only want the most common comorbidities, keep top N
#     top_n_comorbs = 10  # change if you like
#     top_probs = problems_min['problem_desc'].value_counts().head(top_n_comorbs).index
#     problems_min = problems_min[problems_min['problem_desc'].isin(top_probs)]
    
#     # 2. Make wide table: one row per subject, one column per problem_desc (0/1)
#     comorb_df = (
#         pd.crosstab(problems_min['subject'], problems_min['problem_desc'])
#         .clip(upper=1)     # any count >1 becomes 1 (has comorbidity)
#         .reset_index()
#     )


def clean_problems(df, problems_mapping_dict):
    """
    Clean problems table and create one-hot comorbidity features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw problems table.
    problems_config : dict

    Returns
    -------
    pd.DataFrame
        One row per subject, with binary comorbidity columns.
    """

    df = df.copy()

    logging.info('Cleaning column names...')

    df = dct.clean_col_names(df)

    required_cols = ["subject", "problem_desc", "problem_dt_tm"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in problems df: {missing_cols}")

    df = df.drop(columns=["encntr_id"], errors="ignore")

    df = df.dropna(subset=["problem_dt_tm"])
    

    logging.info('Standardising problem_desc column...')

    
    clean_df = dct.map_col_values(
                    df=df,
                    mapping_source= problems_mapping_dict,
                    col="problem_desc",
                    drop_unmapped=True
    )


    problems_min = (
        clean_df[["subject", "problem_desc"]]
        .dropna(subset=["subject", "problem_desc"])
        .drop_duplicates()
    )
    

    comorb_df = (
        pd.crosstab(
            problems_min["subject"],
            problems_min["problem_desc"]
        )
        .clip(upper=1)
        .reset_index()
    )

    
    return comorb_df


def run(cfg_path, out_csv_name = 'problems_clean.csv', save=True, return_df=False, verbose = False):

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
  
    logging.info('Loading the config file...')

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})
    config_cols = cfg.get('columns', {})

    # -----------------------------
    # Load and clean datasets
    # -----------------------------

    logging.info('Loading data...')

    problems_path =  PROJECT_ROOT / paths['problems']
    problem_mapping_path =  PROJECT_ROOT / paths['problem_mapping']
    
    problems_df = pd.read_csv(problems_path)
    
    with open(problem_mapping_path, "r") as f:
        mapping_dict = yaml.safe_load(f)
    
    logging.info(
        f"Loaded: problems = {problems_df.columns}")

    comorbidities_df = clean_problems(problems_df, mapping_dict)

    # ------------------- SAVE ---------------------------------------

    if save:
        logging.info('Saving outputs...')
    
        problems_out_csv = INTERIM_DIR / out_csv_name
        comorbidities_df.to_csv(problems_out_csv, index=False)
    
        logging.info(f"Wrote cleaned problems data to: {problems_out_csv}")
    
        logging.info(f"Final df columns for problems: {problems_df.columns}")
        logging.info(f"Final comorbidities in problems table: {comorbidities_df.columns}")

    if return_df:
        return comorbidities_df


if __name__ == "__main__":
     
    if len(sys.argv) != 3:

        logging.warning("Usage: python clean_problems.py <configs/config.yaml> <problems_out_csv_name.csv>")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    problems_out_csv  = sys.argv[2]

    run(cfg_path, problems_out_csv)