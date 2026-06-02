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
from src.utils import data_merging_tools as dmt
from src.utils import micro_cleaning_tools as mct


# Robust ESBL aggregation (handles bool, 0/1, or strings like "ESBL")
def infer_episode_esbl(x: pd.Series):

    """
    Takes in a pd.Series of all resistance phenotypes isolated during infection episode.

    Returns 'ESBL' if any samples isolated durint the episode were ESBL positive 
    """

    # numeric 0/1
    if pd.api.types.is_numeric_dtype(x):
        return int((x.fillna(0) > 0).any())
    
    # strings / categories
    vals = set(str(v).strip().lower() for v in x.dropna().unique())

    if "esbl" in vals:
        return "esbl"
    else:
        return "non-esbl"


# defining a clinical hierarchy for microbiology sites 

def resolve_site_value(sites:pd.Series):
    """
    Convert a set of sites into a single site string.
    - If empty / missing -> None
    - If size 1 -> that element
    - If size >1 -> choose by hierarchy (lowest numeric value)
    """

    sites_set = set(sites)

    site_hierarchy = {
    # Tier 1: Truly Sterile (Internal)
    "blood": 1,
    "tissue/biopsy": 2,
    
    # Tier 2: Clinically Sterile (Protected)
    "urine": 3,
    "tips_devices": 4,
    
    # Tier 3: Colonized (Exposed to Skin/Environment)
    "wound": 5,
    "drain": 6,
    
    # Tier 4: Heavily Colonized (Normal Flora)
    "sputum": 7,
    "high_vaginal": 8,
    "low_vaginal": 9,
    
    # Tier 5: Miscellaneous
    "other": 10
    }
    
    if sites_set is None:
        return None
    if not isinstance(sites_set, (set, frozenset)):
        # if it's already a single value (string), just return it
        return sites_set

    if len(sites_set) == 0:
        return None
    if len(sites_set) == 1:
        return next(iter(sites_set))

    return min(sites_set, key=lambda s: site_hierarchy.get(s, 999))


        

# SURGERY AGGREGATION - keep closest surgery to infection episode start date 

def infer_episode_surgery(df, infection_idx = 'infection_id', s_start = 'surgery_start_dt', s_stop = 'surgery_stop_dt', flag = ''):

    return


def build_infection_eps(df, group_col = 'infection_id'):

    def unique_sorted_list(x):
        return sorted(pd.Series(x).dropna().astype(str).unique().tolist())

    episodes_df = (
    df.groupby(group_col)
    .agg(
        subject = ('subject', 'first'),
        admission_dt = ('admission_date', 'first')
        discharge_dt = ('discharge_date', 'first'), 
        age_at_admission = ('age_at_admission', 'first'), 
        imdd = ('index_of_multiple_deprivation_decile', 'first)'),
        
        
        site = ('site_grouped', lambda x: resolve_site_value(x)),
        all_sites=('site_grouped', set)
        organism_bug = ('organism_bug', 'first'),
        
        esbl_status = ('esbl_status', lambda x: infer_episode_esbl(x)),
        tfc_desc = ('tfc_desc', set), 
        
        # surgeries

        n_surgeries = ('surgery_start_dt', 'count')
                

        first_culture_dt=('latest_collect_dt', 'min'),
        last_culture_dt=('latest_collect_dt', 'max'),
        n_samples=('lab_test_id', 'count')


           # prophylaxis
        prophylaxis=('prophylaxis', 'first'), # this should already be sorted for each infection ep 
        prophylaxis_admin_dt=('prophylaxis_admin_dt', 'first'),
        delta_hours_prophylaxis=('delta_hours_prophylaxis', 'min'),
        n_prophylaxis_agents=('n_prophylaxis_agents', 'max'),
        is_therapeutic=('is_therapeutic', 'max'),
        tier=('tier', 'min'),
        past_abx = ('past_abx', 'max'),
    )
    .reset_index())

    return episodes_df



  


def run(cfg_path, infection_eps_out_csv = 'infection_eps.csv', save = True, return_df = False, verbose = False):

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


    infection_eps, micro_df = build_infection_eps(final_df)

    esbl_counts = infection_eps['esbl_status'].value_counts()
    logging.info(f" There are {len(infection_eps)} infection episodes. {round((esbl_counts[1]/esbl_counts.sum())*100)}% are ESBL infections.")


    logging.info("Resolving sample sites…")

    infection_eps["sites"] = infection_eps["sites"].apply(resolve_site_value)

    cols_to_keep = ['subject', 'lab_test_id', 'latest_collect_dt', 'latest_received_dt',
       'latest_result_dt',
       'procedure_desc',
        'surgery_start_dt', 'surgery_stop_dt', 'admission_date', 'age_at_admission',
       'discharge_date', 
       'index_of_multiple_deprivation_decile',
       'tfc_desc', 'prophylaxis', 'delta_hours_prophylaxis', 'is_therapeutic',
       'past_abx', 'infection_id', 'site', 'culture_type', 'site_grouped']

    infection_episodes_df = infection_eps.merge(
                            micro_df[cols_to_keep],
                            on = 'infection_id',
                            how = 'left',
                            validate="one_to_many")
    

    # ------------------- SAVE ---------------------------------------

    if save: 
        out_path = INTERIM_DIR / infection_eps_out_csv
        infection_episodes_df.to_csv(out_path, index = False)
    
        logging.info(f"Saved final mcs table with prophylaxis and past abx in {out_path}")

    if return_df:
        return infection_episodes_df
        logging.info('Returned the processed datasets! :)')


if __name__ == '__main__':
    import sys 

    if len(sys.argv) != 3:
        logging.warning("Usage: python get_prophylaxis_past_abx.py <config.yaml> <mcs_output.csv>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    out_csv_name = sys.argv[2]

    run(cfg_path, out_csv_name)
