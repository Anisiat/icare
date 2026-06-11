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


def mode_or_first(series):
    """
    Return most common non-null value.
    If tie, return first encountered mode.
    """
    s = series.dropna()

    if s.empty:
        return np.nan

    return s.mode().iloc[0]


    
def infer_episode_esbl(x: pd.Series):
    """
    Episode-level ESBL rule:
    if any sample in the episode is ESBL-positive, classify the episode as ESBL.
    """
    vals = pd.Series(x).dropna()

    if len(vals) == 0:
        return None

    if pd.api.types.is_numeric_dtype(vals):
        return "ESBL" if (vals > 0).any() else "non-ESBL"

    cleaned = set(str(v).strip().lower() for v in vals.unique())
    return "ESBL" if "esbl" in cleaned else "non-ESBL"


def resolve_site_value(sites: pd.Series):
    """
    Resolve multiple grouped sites into one representative site using a hierarchy.
    """
    site_hierarchy = {
        "blood": 1,
        "urine": 2,
        "tips_devices": 3,
        "drain": 4,
        "wound": 5,
        "sputum": 6,
        "tissue/biopsy": 7,
        "high_vaginal": 8,
        "low_vaginal": 9,
    }

    sites_set = set(pd.Series(sites).dropna().astype(str).str.strip().str.lower())
    
    if len(sites_set) == 0:
        return np.nan

    valid_sites = [s for s in sites_set if s in site_hierarchy]

    if len(valid_sites) == 0:
        return np.nan

    if len(valid_sites) == 1:
        return valid_sites[0]

    return min(valid_sites, key=lambda s: site_hierarchy.get(s, 999))


def unique_sorted_list(x: pd.Series):
    return sorted(pd.Series(x).dropna().astype(str).unique().tolist())


def unique_joined(x: pd.Series):
    vals = sorted(pd.Series(x).dropna().astype(str).unique().tolist())
    return " | ".join(vals) if vals else None


def first_non_null(x: pd.Series):
    x = pd.Series(x).dropna()
    return x.iloc[0] if len(x) > 0 else None


def build_infection_eps(df, group_col='infection_id'):
    """
    Collapse isolate/sample-level microbiology data to one row per infection episode.
    """

    df = df.sort_values([group_col, 'latest_collect_dt'], na_position = 'last').copy()

    episodes_df = (
        df.groupby(group_col)
        .agg(
            # identifiers / demographics / admission context
            subject=('subject', 'first'),
            #spell_identifier=('spell_identifier', 'first'),
            admission_date=('admission_date', 'first'),
            discharge_date=('discharge_date', 'first'),
            age_at_admission=('age_at_admission', 'first'),
            discharge_destination=('discharge_destination', 'first'),
            imd_decile=('index_of_multiple_deprivation_decile', 'first'),
            is_emergency=('is_emergency', 'first'),

            # specialty / service context
            tfc =('tfc_group', mode_or_first),

            # microbiology content
            organism_bug=('organism_bug', first_non_null),

            site=('site_grouped', resolve_site_value), # not using site function here 
            all_sites_grouped=('site_grouped', unique_joined),
            n_sites=('site_grouped', 'nunique'),

            esbl_status=('esbl_status', infer_episode_esbl),

            # microbiology timing
            first_culture_dt=('latest_collect_dt', 'min'),
            last_culture_dt=('latest_collect_dt', 'max'),
            n_samples=('lab_test_id', 'count'),
            
            # surgery context
            first_surgery_start_dt=('surgery_start_dt', 'min'),
            last_surgery_start_dt=('surgery_start_dt', 'max'),
            first_surgery_stop_dt=('surgery_stop_dt', 'min'),
            last_surgery_stop_dt=('surgery_stop_dt', 'max'),
            n_surgeries=('surgery_start_dt', 'nunique'),
            procedure_descs=('procedure_desc', unique_joined),
            surgery_length = ('surgery_length_hours', 'max'),

            # existing episode variables from upstream processing
            infection_ep_start=('infection_ep_start', 'min'),

            # surgery timing relative to infection
            min_days_from_surgery_to_infection=('days_from_surgery_to_infection', 'min'),
            surgery_before_infection=('surgery_before_infection', 'max'),

            # prophylaxis / antibiotic exposure
            prophylaxis=('prophylaxis', unique_joined),
            prophylaxis_group = ('prophylaxis_group', first_non_null),
            prophylaxis_class = ('prophylaxis_class', first_non_null),
            prophylaxis_admin_dt=('prophylaxis_admin_dt', 'first'),
            delta_hours_prophylaxis=('delta_hours_prophylaxis', 'min'),
            n_prophylaxis_agents=('n_prophylaxis_agents', 'max'),
            is_therapeutic=('is_therapeutic', 'max'),
            tier=('tier', 'min'),
            past_abx=('past_abx', 'max'),
        )
        .reset_index()
    )

    # derived columns after aggregation
    episodes_df['culture_span_days'] = (
        episodes_df['last_culture_dt'] - episodes_df['first_culture_dt']
    ).dt.days

    episodes_df['length_of_stay_days'] = (
        episodes_df['discharge_date'] - episodes_df['admission_date']
    ).dt.days

    episodes_df['days_from_admission_to_first_culture'] = (
        episodes_df['first_culture_dt'] - episodes_df['admission_date']
    ).dt.days

    episodes_df['days_from_surgery_to_first_culture'] = (
        episodes_df['first_culture_dt'] - episodes_df['first_surgery_stop_dt']
    ).dt.days

    return episodes_df


# def run(cfg_path, infection_eps_out_csv = 'infection_eps.csv', save = True, return_df = False, verbose = False):

#     logging.getLogger().setLevel(logging.WARNING if not verbose else logging.INFO)
    
#     logging.info('Loading config file...')

#     cfg_path = Path(cfg_path)

#     if not cfg_path.is_absolute():
#         cfg_path = PROJECT_ROOT / cfg_path

#     cfg = dct.load_config(cfg_path)
#     paths = cfg.get('paths', {})
#     config_cols = cfg.get('columns', {})

#     # Load inputs
#     logging.info("Loading input CSVs…")


#     # ------------------- SAVE ---------------------------------------

#     if save: 
#         out_path = INTERIM_DIR / infection_eps_out_csv
#         infection_episodes_df.to_csv(out_path, index = False)
    
#         logging.info(f"Saved final mcs table with prophylaxis and past abx in {out_path}")

#     if return_df:
#         return infection_episodes_df
#         logging.info('Returned the processed datasets! :)')


# if __name__ == '__main__':
#     import sys 

#     if len(sys.argv) != 3:
#         logging.warning("Usage: python get_prophylaxis_past_abx.py <config.yaml> <mcs_output.csv>")
#         sys.exit(1)

#     cfg_path = sys.argv[1]
#     out_csv_name = sys.argv[2]

#     run(cfg_path, out_csv_name)
