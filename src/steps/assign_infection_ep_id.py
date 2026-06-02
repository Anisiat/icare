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
from src.utils import micro_cleaning_tools as mct


def get_infection_ep_id(
    df: pd.DataFrame,
    spell_col: str = "spell_identifier",
    organism_col: str = "organism_bug",
    dt_col: str = "latest_collect_dt",
    site_col: str = "site_grouped",
    esbl_col: str = "esbl_status",
    window_days: int = 14
) -> pd.DataFrame:

    """"
    Takes clean microbiology/surgery table as input and segments infection episodes based on a set of clinical rules.

    Infection episodes were defined as isolation of the same organism within a hospital spell, where consecutive positive cultures were separated by no more than 14 days. 
    
    A new episode was defined if a different organism was isolated or if more than 14 days had elapsed since the previous 
    positive culture for the same organism.
    

    A new infection episode is considered if:
        -  >14 days have elapsed since the previous positive sample of the same organism, not since the first.
        - a new organism is identified at a new site - Same spell can have multiple concurrent infection episodes.

    Returns:
      pd.DataFrame with an issigned infection ID 
    """

    
    required = [spell_col, organism_col, dt_col, site_col, esbl_col]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    micro_df = df.copy()

    # Ensure datetime
    micro_df[dt_col] = pd.to_datetime(micro_df[dt_col], errors="coerce")
    if micro_df[dt_col].isna().any():
        bad = micro_df[micro_df[dt_col].isna()]
        raise ValueError(
            f"Found {bad.shape[0]} rows with invalid {dt_col} after to_datetime(). "
            f"Fix or drop these rows before building episodes."
        )

    # Sort for deterministic episode construction
    micro_df = micro_df.sort_values([spell_col, organism_col, dt_col])

    # this is the group object which holds the set of rules that define how the df should be partitioned
    # within each group the rows have already been sorted by date 
    g = micro_df.groupby([spell_col, organism_col], sort=False)

    # calculates the difference in days between consecutive rows of each group 
    # .diff() computes current row minus previous row so will get NaT values for each first row of each group
    gap_days = g[dt_col].diff().dt.days

    # a new episode will be defined as the first entry of a group or if the gap is bigger than the defined time-window
    #  new_episode will be a boolean series marking which rows are a new episode and which aren't - same length as micro_df
    new_episode = gap_days.isna() | (gap_days > window_days)

    # Episode number within (spell, organism)
    # Groupes the new_episode boolean series and groups it using the index from the micro_df column grouped by spell and organism then 
    # computes the cummulative sum to count episode numbers - how many episodes and which episode does each row belong to 
    micro_df["episode_number"] = new_episode.groupby([micro_df[spell_col],
                                                       micro_df[organism_col]]).cumsum().astype(int)

    # Unique episode id generation 
    # Assigning a unique infection episode ID allows you to group by episode and aggregate all the other info 
    #  E.g. how many samples taken per episode, episode start date, set of sites were samples were collected
    micro_df['infection_id'] = (
        micro_df[spell_col].astype(str) + "|" +
        micro_df[organism_col].astype(str) + "|" +
        micro_df["episode_number"].astype(str)
    )

    # Episode start date = earliest sample date within infection_episode_id
    micro_df["infection_ep_start"] = micro_df.groupby("infection_id")[dt_col].transform("min")

    return micro_df




def run(cfg_path, mcs_out_csv = 'mcs_clean.csv', save = True, return_df = False, verbose = False):

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    logging.info('Loading config file...')

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})
    config_cols = cfg.get('columns', {})

    # Load inputs
    logging.info("Loading input CSVs…")

    mcs_path = PROJECT_ROOT / paths['clean_mcs']
    
    mcs_df = pd.read_csv(mcs_path) 
    mcs_df_ep_id = get_infection_ep_id(mcs_df)


    # ------------------- SAVE ---------------------------------------

    if save: 
        out_path = INTERIM_DIR / mcs_out_csv
        mcs_df_ep_id.to_csv(out_path, index = False)
    
        logging.info(f"Saved final mcs df with assigned ep id in {out_path}")

    if return_df:
        return mcs_df_ep_id
        logging.info('Returned the processed datasets! :)')
        

if __name__ == '__main__':
    import sys 

    if len(sys.argv) != 3:
        logging.warning("Usage: python assign_infection_id.py <configs/config.yaml> <mcs_name.csv>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    out_csv_name = sys.argv[2]

    run(cfg_path, out_csv_name)