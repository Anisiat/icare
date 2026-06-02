import yaml, os
import pandas as pd
import numpy as np 
import re
import logging 
import sys



# -----------------------------
# Report new ESBL numbers
# -----------------------------

def filter_nosocomial(
    df, admission_col = 'admission_date', sample_collect_dt = 'latest_collect_dt', discharge_dt = 'discharge_date', 
    thirtyd_post_discharge = True
):
    """
    Keep only nosocomial infections, defined as samples collected
    at least 48 hours after admission and before discharge.
    """

    df = df.copy()

    df[admission_col] = pd.to_datetime(df[admission_col])
    df[sample_collect_dt] = pd.to_datetime(df[sample_collect_dt])
    df[discharge_dt] = pd.to_datetime(df[discharge_dt])

    df["time_to_sample_collection"] = (
        df[sample_collect_dt] - df[admission_col]
    )

    df["delta_discharge_sample_collection"] =  df[sample_collect_dt] - df[discharge_dt]

    if thirtyd_post_discharge:
        df = df[
            (df["time_to_sample_collection"] >= pd.Timedelta(hours=48)) &
            (df["delta_discharge_sample_collection"] <= pd.Timedelta(days = 30))
        ]
    
    else: 
        df = df[
            (df["time_to_sample_collection"] >= pd.Timedelta(hours=48)) &
            (df[sample_collect_dt] <= df[discharge_date])
        ]

    return df


# -----------------------------
# AST rules (R/S sets)
# -----------------------------

def get_ast_rules(ceph_r:pd.DataFrame, ceph_s:pd.DataFrame):

    """
    Get a set of sensitivities for 3rd generation chephalosporins. 

    MAYBE RETHINK: is the MIC of all ceph rs the same and if not should you make a MIC set for each abx?

    """

    ceph_r = ceph_r.copy()
    ceph_s = ceph_s.copy()

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



# -----------------------------
# ESBL labelling
# -----------------------------

def label_esbl_status(row, ceph_r_set, ceph_s_set):


    # Extract sesnitivity results for each antibiotic
    cefo = row.get('cefotaxime')
    cefta = row.get('ceftazidime')
    ceftr = row.get('ceftriaxone')
    cefpo = row.get('cefpodoxime')
    marker = row.get('esbl markers (ss = present)')

    marker_positive = marker in ['sensitive', 'susceptible', 'susceptible with optimised dosing, refer to antimicrobial policy']
    marker_negative = marker == 'resistant'

    
    # Confirmed ESBL logic (3GCR resistance pattern or ESBL marker = sensitive)
    confirmed_esbl = (
        (cefo in ceph_r_set or cefta in ceph_r_set) or
        (ceftr in ceph_r_set or cefta in ceph_r_set) or
        (cefpo in ceph_r_set) or
        (marker_positive)
    )
    
    # Non-ESBL logic (3GCS sensitivity or ESBL marker = resistant)
    non_esbl = (
        (cefo in ceph_s_set and cefta in ceph_s_set) or 
        (ceftr in ceph_s_set and cefta in ceph_s_set) or
        (cefpo in ceph_s_set) or
        (marker_negative)
    )
    
    if confirmed_esbl:
        return 'ESBL'
    elif non_esbl:
        return 'non-ESBL'
    else:
        return 'unknown'  # fallback for ambiguous or missing data


# -----------------------------
# Drop unneccessary columns 
# -----------------------------

def drop_columns(df, cfg):
    """
    Drops unnecessary columns, including AST-related columns
    after inferring the ESBL labels.
    """
    # Normalize names for consistent matching
    df.columns = (
        df.columns
        .str
        .lower()
        .str
        .strip()
        .str
        .replace(' ', '_')
    )

    cols_to_drop = cfg["drop_columns"]["general"]
    ast_keywords = cfg["drop_columns"]["ast_keywords"]

    # Drop AST columns only if ESBL status exists
    if "esbl_status" in df.columns:
        ast_to_drop = [
            col for col in df.columns
            if any(k in col.lower() for k in ast_keywords)
        ]

        if ast_to_drop:
            print(f"Dropping AST columns: {ast_to_drop}")
            df = df.drop(columns=ast_to_drop)
    else:
        print("esbl_status not found — skipping AST column drop.")

    # Drop general metadata columns
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df = df.drop_duplicates()

    return df



# -----------------------------
# Resolve conflicting ESBL status
# -----------------------------

def resolve_conflicting_esbl_status(df, esbl_col = 'esbl_status', lab_id_col = 'lab_test_id'):
    """
    Aggregates rows by lab ID, checks if any organism in that sample is ESBL-positive,
    and updates the ESBL status for ALL rows belonging to that lab ID in the original DataFrame.

    So even if one of the bugs in the sample is susceptible to 3GCs, the sample will still be labelled as ESBL
    bc there's an ESBL-producing organism in the sample. 

    Args:
        df (pd.DataFrame): The original DataFrame (e.g., final_df).
        esbl_col (str): The name of the column containing the ESBL status (e.g., 'esbl_status').
        lab_id_col (str): The name of the column containing the lab sample ID (e.g., 'lab_test_id').

    Returns:
        pd.DataFrame: The original DataFrame with the updated ESBL status column.
    """

    # 1. Group the DataFrame by lab ID and aggregate the ESBL statuses into a list/set
    
    aggregated_df = df.groupby(lab_id_col)[esbl_col].agg(list).reset_index()

    # 2. Identify all lab IDs where 'ESBL' is present in the aggregated list of statuses.
    
    esbl_ids= [
    row[lab_id_col]
    for index, row in aggregated_df.iterrows()
    if 'ESBL' in row[esbl_col]]

    # Create a boolean mask: True for rows whose lab ID is in the esbl_ids list
    mask = df[lab_id_col].isin(esbl_ids)

    df.loc[mask, esbl_col] = 'ESBL'

    return df

# -----------------------------
# Report new ESBL numbers
# -----------------------------

def report_esbl_numbers(df, esbl_col = 'esbl_status', lab_id_col = 'lab_test_id' ):
    """
    Reports how many ESBL and non-ESBL samples there are (unique lab IDs).
    """
   
    if esbl_col not in df.columns:
        print(f"Column '{esbl_col}' not found in dataframe.")
        return

    # group by lab_id to avoid double-counting the same test
    esbl_counts = (
        df.groupby(lab_id_col)[esbl_col]
        .first()  # take the first value if duplicates
        .value_counts(dropna=False)
    )

    print("ESBL status counts:")
    for status, count in esbl_counts.items():
        print(f"  {status}: {count}")

    print(f"\nTotal unique samples: {esbl_counts.sum()}")



# ========================================
# Group the microbiology site values 
# ========================================


def group_specimen_site(
    df,
    site_col = "site",
    culture_type_col = "culture_type",
    abdomen_spec = False, 
    drop_original_cols = False,
) -> pd.DataFrame:
    """
    Clean and group microbiology specimen site into clinically meaningful categories.

    Infers `site_grouped` from free-text `site` and `culture_type`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    site_col : str, default="site"
        Column containing specimen site free text.
    culture_type_col : str, default="culture_type"
        Column containing culture type free text.
    abdomen_spec : bool, default=False
        If True, creates a more specific "abdomen wound" category.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with a new column: `site_grouped`.
    """

    df = df.copy()
    df["site_grouped"] = pd.NA

    # Clean and normalise text
    def _clean_text(series) -> pd.Series:
        return (
            series.astype("string")
            .str.lower()
            .str.strip()
            .str.replace("-", " ", regex=False)
            .str.replace("/", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )

    s = _clean_text(df[site_col])
    ct = _clean_text(df[culture_type_col])

    # Combine both sources for easier matching
    combined = (s.fillna("") + " " + ct.fillna("")).str.strip()

    # Helper for assignment without overwriting earlier, more specific matches

    def assign_group(mask, label: str) -> None:
    
        df.loc[df["site_grouped"].isna() & mask, "site_grouped"] = label
    
        # ---- Define patterns ----
        # Word boundaries (\b) are useful when you want the actual word, not part of another word.
    blood_pat = r"\bblood\b"
    urine_pat = r"\burine\b|\burin\b"
    sputum_pat = r"\bsputum\b|\bbronchoalveolar\b|\bbronchial\b|\bbal\b"
    wound_pat = r"\bwound\b|\bpus\b"
    tissue_pat = r"\btissue\b|\bbiopsy\b"
    drain_pat = r"\bdrain\b|\bfluid\b|\bbile\b|(?=.*\bpleural\b)(?=.*\bfluid\b)"
    abdomen_pat = r"\babdomen\b|\babdominal\b"
    high_vaginal_pat = r"\bhigh vaginal\b"
    low_vaginal_pat  = r"\blow vaginal\b"
    tips_devices_pat =r"\btips\b|\btip\b|\bdevices\b|\bcatheter\b"

    # ---- More specific rules first ----
    if abdomen_spec:
        abdomen_wound = combined.str.contains(wound_pat, na=False) & combined.str.contains(abdomen_pat, na=False)
        assign_group(abdomen_wound, "abdomen wound")

        abdomen_drain = combined.str.contains(drain_pat, na=False) & combined.str.contains(abdomen_pat, na=False)
        assign_group(abdomen_drain, "abdomen drain")

    
    blood = combined.str.contains(blood_pat, na=False)
    assign_group(blood, "blood")

    urine = combined.str.contains(urine_pat, na=False)
    assign_group(urine, "urine")

    sputum = combined.str.contains(sputum_pat, na=False)
    assign_group(sputum, "sputum")
    
    wound = combined.str.contains(wound_pat, na=False)
    assign_group(wound, "wound")
    # Drain/fluid after pleural, because pleural fluid should usually stay pleural
    drain = combined.str.contains(drain_pat, na=False)
    assign_group(drain, "drain")
    
    tissue = combined.str.contains(tissue_pat, na=False)
    assign_group(tissue, "tissue/biopsy")

    tips_devices = combined.str.contains(tips_devices_pat, na=False)
    assign_group(tips_devices, "tips_devices")

    # low_vaginal = combined.str.contains(low_vaginal_pat, na=False)
    # assign_group(low_vaginal, "low_vaginal")

    # high_vaginal = combined.str.contains(high_vaginal_pat, na=False)
    # assign_group(high_vaginal, "high_vaginal")


    # Fallback
    df["site_grouped"] = df["site_grouped"].fillna("other")

    if drop_original_cols:
        df = df.drop(['site', 'culture_type'], axis = 1)

    return df


# ========================================
# separate mcs from microbiology screens
# ========================================


def split_microbiology_screens(
                        microbiology_df,
                        screen_codes = ['crocul', 'itucul', 'rgns', 'xincul'],
                        screens_df_cols = ['subject', 'admission_date', 'lab_test_id', 'latest_collect_dt', 'latest_received_dt', 'latest_result_dt', 'culture_type','site', 'organism_bug', 'culture_code', 'esbl_status']
):
    """
    Split a microbiology table into:
      - microbiology_clean_df: all non-screen (MCS) rows
      - microbiology_screens_df: only screen rows with a reduced set of columns

    Parameters
    ----------
    microbiology_df : pd.DataFrame
        Full microbiology table.
    screen_codes : iterable of str
        Values of `culture_code_col` that correspond to screening cultures.
    screens_df_cols : sequence of str
        Columns to keep in the screens dataframe.
    culture_code_col : str
        Name of the column that stores the culture code.

    Returns
    -------
    microbiology_clean_df, microbiology_screens_df : (pd.DataFrame, pd.DataFrame)
    """

    # check if patients had a screen 
    all_microbiology = microbiology_df.copy()

    required_cols = ['culture_code', 'latest_collect_dt', 'surgery_start_dt']
    
    for c in required_cols:
        if c not in all_microbiology.columns:
            raise KeyError(f"Expected column {c} not found in microbiology_df")

    # ensure date time 

    all_microbiology['latest_collect_dt'] = pd.to_datetime(all_microbiology['latest_collect_dt'])

    all_microbiology['surgery_start_dt'] = pd.to_datetime(all_microbiology['surgery_start_dt'])

    # only keep screens collected before surgery

    mask_screen = all_microbiology['culture_code'].isin(screen_codes)

    mask_preop_screen = mask_screen & ( all_microbiology['latest_collect_dt'] < all_microbiology['surgery_start_dt'])

    microbiology_screens_df = all_microbiology.loc[mask_preop_screen, screens_df_cols].copy()
    microbiology_screens_df = microbiology_screens_df.copy().drop(columns=["culture_code"], axis=1)
    microbiology_clean_df   = all_microbiology.loc[~mask_screen].copy()
    microbiology_clean_df = microbiology_clean_df.copy().drop(columns=["culture_code"], axis=1)
    
    return microbiology_clean_df, microbiology_screens_df
    
    



    





