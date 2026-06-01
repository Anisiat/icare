import yaml, os
import pandas as pd
import numpy as np 
import re
import logging 



# clean df columns 

def clean_df_columns(df:pd.DataFrame, drop_cols = ('cefixime', 'ceftolozane-tazobactam'),  datetime_keywords = ('dt', 'date', 'time')):
    
    df_out = df.copy()

    # normalise column names 
    def _normalise(name:str) -> str:
        name = name.replace("'", "").replace('"', '').strip().lower()
        name = re.sub(r"\s+", "_", name) #this means replace spaces (\s is space and then + means one or more)
        name = re.sub(r"[0-9a-zA_Z]", "_", name) # replace any character that is no 0-9, a-z or A-Z
        name = re.sub(r"__+", "_", name).strip("_")
        return name 

    old_cols = df_out.columns
    new_columns = [_normalise(c) for c in df_out.columns]
    df_out.columns = new_cols 
    renamed_cols = dict(zip(old_cols, new_cols))
    
    logging.info(f"Renamed columns: {renamed_cols}")

    # drop useless columns 
    to_drop = [c for c in df_out.columns if c in drop_cols]

    if to_drop:
        df_out = df_out.drop(to_drop)
        logging.info(f"dropped columns {to_drop}")

    # convert date columns to datetime

    for col in df_out.columns:
        if any(k.lower() in col for k in datetime_keywords):
            try:
                df_out[col] =  pd.to_datetime(df_out[col, errors = 'coerce')
                logging.info(f"Converted {col} to datetime")
            except Exception as e: 
                logging.warning(f"Could not convert {col}: {e}")
    
    
    logging.info("Column cleaning finished")
    return df


def get_ast_rules(ceph_r:pd.DataFrame, ceph_s:pd.DataFrame):

    """
    Get a set of sensitivities for 3rd generation chephalosporins. 

    MAYBE RETHINK: is the MIC of all ceph rs the same and if not should you make a MIC set for each abx?

    """

    try:
        ceph_r[['antibiotic', 'sensitivity']] = ceph_r['SENSITIVITY'].str.split(":", n = 1, expand = True)
        ceph_r.drop(columns = 'SENSITIVITY', inplace=True)

        ceph_s[['antibiotic', 'sensitivity']] = ceph_s['SENSITIVITY'].str.split(":", n = 1, expand = True)
        ceph_s.drop(columns = 'SENSITIVITY', inplace=True)

    except KeyError as e:
        logging.error(f"Could not get ast rules: {e}")

    # strip trailing whitespaces from all columns after splitting 

    ceph_s = ceph_s.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    ceph_r = ceph_r.apply(lambda col: col.str.strip() if col.dtype == 'object' else col) 

    # converting sensitivities into a set for speed 
    ceph_r_set = set(ceph_r['sensitivity'])
    ceph_s_set = set(ceph_s['sensitivity'])

    return ceph_r_set, ceph_s_set 


# ESBL inferring and labeling function

def label_esbl_status(row):

    try:
        # Extract sesnitivity results for each antibiotic
        cefo = row.get('cefotaxime')
        cefta = row.get('ceftazidime')
        ceftr = row.get('ceftriaxone')
        cefpo = row.get('cefpodoxime')
        marker = row.get('esbl markers (ss = present)')

    except ValueError as e:

    
    # Confirmed ESBL logic (3GCR resistance pattern or ESBL marker = sensitive)
    confirmed_esbl = (
        (cefo in ceph_r_set or cefta in ceph_r_set) or
        (ceftr in ceph_r_set or cefta in ceph_r_set) or
        (cefpo in ceph_r_set) or
        (marker in ['sensitive', 'susceptible', 'susceptible with optimised dosing, refer to antimicrobial policy'])
    )
    
    # Non-ESBL logic (3GCS sensitivity or ESBL marker = resistant)
    non_esbl = (
        (cefo in ceph_s_set and cefta in ceph_s_set) or 
        (ceftr in ceph_s_set and cefta in ceph_s_set) or
        (cefpo in ceph_s_set) or
        (marker == 'resistant')
    )
    
    if confirmed_esbl:
        return 'ESBL'
    elif non_esbl:
        return 'non-ESBL'
    else:
        return 'unknown'  # fallback for ambiguous or missing data


def relabel_or_drop_unknowns(df,
                             ceph_r_set,
                            label_col = 'esbl_status'):
    
    unknown_idx = df[df[label_col] == 'unknown'].index

    esbl_count = 0
    non_esbl_count = 0
    idx_to_drop =[]

    for i in unknown_idx:
    
        if df.loc[i, ['cefotaxime', 'cefpodoxime', 'ceftazidime', 'ceftriaxone', 'esbl markers (ss = present)']].isna().sum() == 5:
            idx_to_drop.append(i)

        elif any(col in ceph_r_set for col in list(df.loc[i,['cefotaxime', 'cefpodoxime', 'ceftazidime', 'ceftriaxone']])):
            df.loc[i,'esbl_status'] = 'ESBL'
            esbl_count += 1
        else:
            df.loc[i,'esbl_status'] = 'non-ESBL'
            non_esbl_count += 1
    
    df = df.drop(idx_to_drop)
    
    logging.info(f"Relabeled {esbl_count} as ESBL \nRelabeled {non_esbl_count} as non-ESBL \nDropped {len(idx_to_drop}")

    return df



def load_config(cfg_path="config.yaml"):
    """Read YAML and make paths absolute relative to the config file."""
    cfg_file = Path(cfg_path).resolve()
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    base = cfg_file.parent
    # resolve paths
    cfg["paths"] = {k: str((base / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg

# CLEAN THE DATASET 

def main(cfg_path="../config.yaml"):
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # 1) Load data
    surgery_microbiology = pd.read_csv(cfg['path']['surgery_microbiology'])
    ceph_r = pd.read_csv(cfg['path']['_3gcr'])
    ceph_s = pd.read_csv(cfg['path']['_3gcs']

    logging.info(f"Loaded surgery_microbiology: {surgery_microbiology.shape}, ceph_r: {ceph_r.shape}, ceph_s: {ceph_s.shape}")

    # 2) Build rules (sets of “R” and “S” tokens you’ll check against)
    ceph_r_set, ceph_s_set = get_ast_rules(ceph_r, ceph_s)

    # 3) Label each row for ESBL status 
    surgery_microbiology["esbl_status"] = surgery_microbiology.apply(
        lambda row: label_esbl_status(row, ceph_r_set, ceph_s_set),
        axis=1
    )

    # 4) Fix unknowns: re-label or drop
    surgery_microbiology, rpt = relabel_or_drop_unknowns(
        surgery_microbiology,
        ceph_r_set=ceph_r_set,),
    )


if __name__ = "__main__":
    main("config.yaml")





