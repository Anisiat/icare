import pandas as pd
import logging
from pathlib import Path
import sys


# GET FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'
DATA_DIR = PROJECT_ROOT / 'data'
INTERIM_DIR = DATA_DIR / 'interim'

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# import step modules
from steps.clean_problems import run as clean_problems
from steps.clean_demographics import run as clean_demographics

from steps.clean_microbiology import run as clean_microbiology
from steps.assign_infection_ep_id import get_infection_ep_id as assign_infection_id
from steps.get_abx_exposure import get_abx_exposure as get_abx
from steps.build_ep_surgery_features import get_surgery_features as get_surgery_features
from steps.build_infection_eps import build_infection_eps as build_infection_eps
from steps.infer_comorbidities import update_comorbidities
from steps.get_healthcare_exposure import get_healthcare_exposure
from steps.get_temp_crp import get_temperature
from steps.get_temp_crp import get_crp
from utils import data_cleaning_tools as dct
from steps.get_colonisation import get_colonisation

def run_steps(cfg_path = CONFIG_PATH, verbose = True):

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
        
    logging.info('Loading the config file...')

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})
    config_cols = cfg.get('columns', {})

    logging.info('Cleaning datasets...')

    # 1. Clean source tables
    demographics_df = clean_demographics(cfg_path, save=True, return_df=True)

    #problems df should be loaded as binary encoded comorbidities per subject 
    problems_df = clean_problems(cfg_path, save=True, return_df=True)
    
    # 2. Build microbiology infection episodes
    mcs_df_no_id, screens_df = clean_microbiology(cfg_path, save=True, return_df=True)
          
    mcs_id_df = assign_infection_id(mcs_df_no_id)
    
    # 3. Add antibiotic and surgery exposure
    mcs_id_abx_df = get_abx(mcs_df=mcs_id_df, cfg_path=cfg_path)
    mcs_id_abx_surgery_df = get_surgery_features(mcs_id_abx_df)

    print(mcs_id_abx_surgery_df['esbl_status'].value_counts())

    print(mcs_id_abx_surgery_df.groupby(by = 'lab_test_id')['esbl_status'].value_counts())
    
    # 4. Build final infection episode table
    infection_eps = build_infection_eps(mcs_id_abx_surgery_df)

    logging.info('Merging datasets to get final analysis_df...')
        
    # merge demographics and comorbidities
    analysis_df = (
        infection_eps
        .merge(problems_df, on='subject', how='left')
        .merge(demographics_df, on='subject', how='left')
    )

    # where a subject had no matching comorbidities, fill 0

    comorb_cols = problems_df.columns.drop('subject')
    analysis_df[comorb_cols] = analysis_df[comorb_cols].fillna(0).astype(int)


    # infer comorbidities from prescriptions then drop <0.1
    analysis_df = update_comorbidities(analysis_df, cfg_path = cfg_path )

    # drop comorbidities that have a <10% occurance in the cohort 

    prevalence = analysis_df[comorb_cols].mean()
    low_prev_cols = prevalence[prevalence < 0.1].index.tolist()

    analysis_df = analysis_df.drop(columns = low_prev_cols)
    
    # get prior healthcare exposure in the past 90d

    analysis_df = get_healthcare_exposure(analysis_df, cfg_paths = paths, exposure_type = 'any')
    
    logging.info('getting temperature and crp data')
    
    analysis_df = get_temperature(analysis_df, cfg_path = cfg_path)

    analysis_df = get_crp(analysis_df, cfg_path = cfg_path)

    logging.info('getting colonisation data')
    analysis_df = get_colonisation(analysis_df)


    analysis_df['death_date'] = pd.to_datetime(analysis_df['death_date'])
    analysis_df['discharge_date'] = pd.to_datetime(analysis_df['discharge_date'])
    
    analysis_df['in_hosp_mortality'] = (
        analysis_df['death_date'].notna() & (analysis_df['death_date']<= analysis_df['discharge_date'])
                                       ).astype(int)

    analysis_df_path = DATA_DIR/ 'interim'/ 'analysis_df.csv'
    analysis_df.to_csv(analysis_df_path, index = False)


if __name__ == '__main__':
    
     # Expect: python clean_and_save_dfs.py <config.yaml>
    if len(sys.argv) < 1:
        logging.warning("Usage: python clean_and_save_dfs.py <config.yaml>")
        sys.exit(1)
    
    cfg_path = sys.argv[1]

    run_steps(
    cfg_path=CONFIG_PATH,
    verbose=True
    )

    print("\nPipeline completed successfully.")


    
