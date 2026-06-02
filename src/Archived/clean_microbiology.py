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
from src.utils import surgery_mapping_tools as smt


def run(cfg_path, mcs_out_csv = 'mcs_clean.csv', screens_out_csv = 'screens_clean.csv', save=True, return_df=False, verbose = False):

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
        
    logging.info('Loading the config file...')

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})
    config_cols = cfg.get('columns', {})

    # Load inputs
    logging.info("Loading input CSVs…")

    ceph_r_path = PROJECT_ROOT / paths['_3gcr']
    ceph_s_path = PROJECT_ROOT / paths['_3gcs']
    surgery_microbiology_path = PROJECT_ROOT / paths['surgery_microbiology']

    surgery_microbiology_df = pd.read_csv(surgery_microbiology_path)
    ceph_r = pd.read_csv(ceph_r_path)
    ceph_s = pd.read_csv(ceph_s_path)

    logging.info(f"Loaded: surgery={surgery_microbiology_df.shape}, R={ceph_r.shape}, S={ceph_s.shape}")
    
    # Clean, label, fix unknowns
    logging.info("Cleaning column names & datetimes…")
    surgery_microbiology_df = dct.clean_df_columns(surgery_microbiology_df)

    logging.info("Building R/S rule sets…")
    ceph_r_set, ceph_s_set = mct.get_ast_rules(ceph_r, ceph_s)

    logging.info("Labelling ESBL status…")
    surgery_microbiology_df['esbl_status'] = surgery_microbiology_df.apply(
        lambda row: mct.label_esbl_status(row, ceph_r_set, ceph_s_set),
        axis=1
    )

    logging.info("Relabelling or dropping unknowns…")
    surgery_microbiology_df = mct.relabel_or_drop_unknowns(surgery_microbiology_df, ceph_r_set)

    logging.info("Dropping unneccessary columns...")
    surgery_microbiology_df = mct.drop_columns(surgery_microbiology_df, cfg)

    logging.info('Resolving conflicting ESBL statuses...')
    surgery_mircobiology_df = mct.resolve_conflicting_esbl_status(surgery_microbiology_df)

    mct.report_esbl_numbers(surgery_mircobiology_df)

    logging.info('Splitting mcs from screens...')

    surgery_mcs_df, surgery_screens_df = mct.split_microbiology_screens(surgery_microbiology_df)

    logging.info('Grouping microbiology sites...')

    surgery_mcs_df = mct.group_specimen_site(surgery_mcs_df)
    surgery_screens_df = mct.group_specimen_site(surgery_screens_df)

    
    # SURGERY MAPPING 
    
    logging.info('Loading surgery mappping datasets...')

    tfc_path =  PROJECT_ROOT / paths['tfc_mapping']
    main_specialty_path = PROJECT_ROOT / paths['msc_mapping']

    tfc_df = pd.read_csv(tfc_path)
    main_specialty_df = pd.read_csv(main_specialty_path)

    logging.info('Cleaning mapping datasets...')

    tfc_df = smt.clean_mapping_df(tfc_df)
    main_specialty_df = smt.clean_mapping_df(main_specialty_df)

    logging.info('Converting mapping df to dictionaries...')

    tfc_mapping = smt.convert_df_to_dict(tfc_df, value_col='treatment_function')
    main_specialty_mapping = smt.convert_df_to_dict(main_specialty_df, value_col='main_specialty')

    logging.info('Mapping tfc/msc codes in maicrobiology df...')
    
    surgery_mcs_df = smt.map_codes(surgery_mcs_df, tfc_mapping, col = 'treatment_function_code', new_col= 'tfc_desc')
    surgery_mcs_df = smt.map_codes(surgery_mcs_df, main_specialty_mapping, col = 'main_specialty_code', new_col= 'main_specialty')

    logging.info('Resolving conflicting surgery codes...')

    surgery_mcs_df = smt.resolve_conflict_surgeries(surgery_mcs_df, 'tfc_desc')

    
    
    # ------------------- SAVE ---------------------------------------
    if save:
    
        mcs_out_csv = INTERIM_DIR / mcs_out_csv
        surgery_mcs_df.to_csv(mcs_out_csv, index=False)
        logging.info(f"Done. Wrote cleaned mcs data to: {mcs_out_csv}")
    
        screens_out_csv = INTERIM_DIR / screens_out_csv
        surgery_screens_df.to_csv(screens_out_csv, index=False)
        logging.info(f"Done. Wrote cleaned screens data to: {screens_out_csv}")
    
        logging.info(f"Final df columns for mcs: {surgery_mcs_df.columns}")

    if return_df:
        return surgery_mcs_df, surgery_screens_df
        logging.info('Returned the processed datasets! :)')

if __name__ == "__main__":
     # Expect: python data_cleaning.py <config.yaml> <output.csv>
    if len(sys.argv) < 4:
        logging.warning("Usage: python clean_microbiology.py <config.yaml> <mcs_output.csv> <screens_output.csv")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    mcs_out_csv  = sys.argv[2]
    screens_out_csv  = sys.argv[3]

    run(cfg_path, mcs_out_csv, screens_out_csv)

