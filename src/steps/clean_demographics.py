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

import warnings
warnings.filterwarnings("ignore", message="Could not infer format")


logging.info(f"current root directory set to {PROJECT_ROOT}")

logging.info(f"Will save outputs to {INTERIM_DIR}")


def run(cfg_path, out_csv_name = 'demographics_clean.csv', save=True, return_df=False, verbose = False):

    """
    Clean the raw demographics dataset and map ethnicity codes using a
    config-defined workflow. Loads input files specified in the YAML config,
    standardizes column names, applies lookup mappings, and keeps only the
    configured final columns.

    Parameters
    ----------
    cfg_path : str or Path
        Path to YAML config defining input paths and selected columns.
    out_csv_name : str
        Filename for saving the cleaned dataset under data/interim/.
    save : bool, default=True
        If True, save the cleaned dataset to disk.
    return_df : bool, default=False
        If True, return the cleaned DataFrame.

    Returns
    -------
    pd.DataFrame or None
        Cleaned demographics dataset if return_df=True, else None. 
        COLS: subject, ethnicity_desc, gender 
    """

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

    logging.info("Loading datasets...")
    

    demographics_path = PROJECT_ROOT / paths['demographics']
    ethnicity_lookup_path = PROJECT_ROOT / paths['ethnicity_lookup']

    demographics_df = pd.read_csv(demographics_path)
    ethnicity_lookup = pd.read_csv(ethnicity_lookup_path)

    logging.info("Cleaning demographics...")

    demographics_df = dct.clean_col_names(demographics_df)

    logging.info("Mapping ethnicity in demographics df...")

    ethnicity_lookup = dct.clean_col_names(ethnicity_lookup)

    ethnicity_dict = dct.map_to_dict(ethnicity_lookup, value_col= 'ethnicity_desc', key_col='ethnicity')

    demographics_df = dct.map_codes(demographics_df, ethnicity_dict, col = 'ethnicity', new_col= 'ethnicity_desc')

    logging.info(f"Keeping demographics columns: {config_cols['demographics']}")

    demographics_df = demographics_df[config_cols['demographics']]

   
    demographics_df = demographics_df[['subject', 'gender', 'death_date', 'ethnicity_desc']]
    
    ethnicity_map = {
        
        # other
        "other - any other ethnic group": "other_unknown",
        "other - not stated": "other_unknown",
        "other - not known": "other_unknown",
        "not known": "other_unknown", 
    
    
           # White
        "white - british": "white",
        "white - any other white background": "white",
        "white - irish": "white",
    
        # Asian
        "asian or asian british - indian": "asian",
        "asian or asian british - pakistani": "asian",
        "asian - any other asian background": "asian",
        "asian or asian british - bangladeshi":"asian",
        "other - chinese" : "asian",
    
        # Black
        "black or black british - african": "black",
        "black or black british - caribbean": "black",
        "black - any other black background": "black",
    
    }
    
    
    demographics_df["ethnicity_desc"] = (
        demographics_df["ethnicity_desc"]
        .replace(ethnicity_map)
        .fillna("other")
    )
    
    demo_counts = demographics_df['ethnicity_desc'].value_counts().reset_index()
    sparse_demo = demo_counts[demo_counts['count'] < 30]['ethnicity_desc']
    
    demographics_df.loc[
        demographics_df['ethnicity_desc'].isin(sparse_demo),
        'ethnicity_desc'
    ] = pd.NA
       


    # ------------------- SAVE ---------------------------------------

    logging.info('Saving clean dataset...')

    # Define output path inside datasets

    if save:
        
        out_csv = INTERIM_DIR/ out_csv_name
        demographics_df.to_csv(out_csv, index=False)
        
        logging.info(f"Done. Wrote cleaned data to: {out_csv}")

    if return_df:
        return demographics_df


if __name__ == "__main__":
     # Expect: python data_cleaning.py <config.yaml> <output.csv>
    if len(sys.argv) < 3:
        logging.warning("Usage: python clean_demographics.py <config.yaml> <clean_demographics.csv>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    out_csv_name  = sys.argv[2]

    run(cfg_path, out_csv_name )