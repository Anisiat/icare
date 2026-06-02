import numpy as np 
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
INTERIM_DIR = DATA_DIR / 'interim'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import data_cleaning_tools as dct

abx_cfg = dct.load_config("../configs/antibiotics.yaml")


def filter_antibiotics(df, medication_col='medication_name_short', abx_cfg = abx_cfg):

    # convert all entries into str to handle Nans
    df[medication_col] = df[medication_col].astype(str)

    # Create boolean masks for filtering
    ends_with_suffix = df[medication_col].str.endswith(tuple(abx_cfg['NON_ANTIBIOTIC_SUFFIXES']))
    in_specific_exclude = df[medication_col].isin(abx_cfg['ABX_TO_EXCLUDE'])
    in_keep = df[medication_col].isin(abx_cfg['ABX_TO_KEEP'])

    # Combine masks: exclude if (ends with suffix OR in specific exclude) AND NOT in keep
    exclude_mask = (ends_with_suffix | in_specific_exclude) & ~in_keep

    abx_administrations = df[~exclude_mask].copy()

    return abx_administrations


def get_past_abx(df,
                 delta_days_past_abx = 90,
                 idx_col  = 'infection_id', 
                 admission_col = 'admission_date', 
                administration_col = 'administration_dt'):

    """
    Compute a binary feature (past_abx) per infection_id depending on whether the 
    patient received antibiotics between 1 and 'delta_days_past_abx' days
    prior to admission.

    Returns
    --------
    df -> one row per idx_col with a column of past_abx (0/1)

    """

    admit_dates = pd.to_datetime(df[admission_col])
    admin_dates = pd.to_datetime(df[administration_col])

    # compute days between past administration and admission (positive = before admission) 
    delta_days = (admit_dates - admin_dates).dt.days
    
    # Keep rows with administration in the chosen time window 
    mask_window = (
        (delta_days >= 1) & (delta_days <= delta_days_past_abx)
    )

    # get lab_id with past abx in the chosen window 
    samples_with_abx = df.loc[mask_window, idx_col].dropna().unique()

    # assign 'past_abx' col directly to the passed dataframe
    df['past_abx'] = df[idx_col].isin(samples_with_abx).astype(int)
    
    return df



def infer_prophylaxis(
    df: pd.DataFrame,
    therapeutic_agents: List[str],
    surgery_start_col: str = "surgery_start_dt",
    administration_dt_col: str = "administration_dt",
    idx_col: str = "infection_id",
    abx_col: str = "medication_name_short",
    route_col: str = "route",
    admit_date_col: str = "admission_date",
    discharge_date_col: str = "discharge_date",
    hours_before_surgery: int = 12,
    hours_after_surgery: int = 1,
    broad_time_window: float = 12.0,
) -> pd.DataFrame:
    """
    Infer prophylaxis per infection_id.

    input df -> should be the mcs_df merged with the antibiotic administrations table 

    Pass 1:
        IV administrations in the peri-operative window
        [surgery_start - hours_before_surgery, surgery_start + hours_after_surgery]

    Pass 2:
        If nothing found for that infection_id, use IV administrations during admission
        within +/- broad_time_window of surgery.

    Preference:
        - prophylactic (non-therapeutic) agents before therapeutic agents
        - closest administration time to surgery
        - if multiple agents share the same chosen administration time, combine them
    """

    required = [
        surgery_start_col,
        administration_dt_col,
        idx_col,
        abx_col,
        route_col,
        admit_date_col,
        discharge_date_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out_df = df.copy()

    # datetime handling
    for col in [surgery_start_col, administration_dt_col, admit_date_col, discharge_date_col]:
        out_df[col] = pd.to_datetime(out_df[col], errors="coerce")

    # normalise strings
    out_df[route_col] = out_df[route_col].astype(str).str.strip().str.lower()
    out_df[abx_col] = out_df[abx_col].astype(str).str.strip().str.lower()

    # IV only
    out_df = out_df[out_df[route_col] == "iv"].copy()

    # delta in hours: positive means before surgery
    out_df["delta_hours_prophylaxis"] = (
        out_df[surgery_start_col] - out_df[administration_dt_col]
    ).dt.total_seconds() / 3600.0
    out_df["abs_delta"] = out_df["delta_hours_prophylaxis"].abs()

    # therapeutic flag
    therapeutic_set = {a.strip().lower() for a in therapeutic_agents}
    out_df["is_therapeutic"] = out_df[abx_col].isin(therapeutic_set)

    # pass 1 mask: IV administration within peri-op window
    pass1_mask = (
        (out_df["delta_hours_prophylaxis"] >= -hours_after_surgery)
        & (out_df["delta_hours_prophylaxis"] <= hours_before_surgery)
    )

    # pass 2 mask: broader window during admission
    pass2_mask = (
        (out_df["abs_delta"] <= broad_time_window)
        & (out_df[administration_dt_col] >= out_df[admit_date_col])
        & (out_df[administration_dt_col] <= out_df[discharge_date_col])
    )

    # assign tiers to the administration df - tier 1 if in peri-op window/ tier 2 if in broad window
    out_df["tier"] = np.where(pass1_mask, 1, np.where(pass2_mask, 2, 'no_prophylaxis'))

    # drop rows where administration is in neither time windows
    # keeps 
    candidates = out_df.dropna(subset=["tier"]).copy()

    all_ids = df[[idx_col]].drop_duplicates()

    if candidates.empty:
        return all_ids.assign(
            prophylaxis="no_prophylaxis",
            delta_hours_prophylaxis=pd.NA,
            is_therapeutic=pd.NA,
            tier=pd.NA,
        )

    # sort so best rows come first:
    # lower tier first, prophylactic before therapeutic, closest to surgery first
    candidates = candidates.sort_values(
        [idx_col, "tier", "is_therapeutic", "abs_delta", administration_dt_col],
        ascending=[True, True, True, True, True],
    )

    # best row per infection_id defines chosen tier and chosen administration time
    # defines the chosen administration time to use as the anchor for prophylaxis
    first_candidate_row = (
        candidates.groupby(idx_col, as_index=False)
        .first()[[idx_col, "tier", administration_dt_col, "delta_hours_prophylaxis", "is_therapeutic"]]
        .rename(columns={administration_dt_col: "prophylaxis_admin_dt"})
    )

    # merge the first row timestamp onto candidates so you know what the first prophylaxis time is
    merged = candidates.merge(
    first_candidate_row[[idx_col, "prophylaxis_admin_dt"]],
    on=idx_col,
    how="inner",
)
    
    # get time difference in minutes so you can include prophylaxis combination 
    merged["time_diff_minutes"] = (
    (merged[administration_dt_col] - merged["prophylaxis_admin_dt"])
    .dt.total_seconds()
    .abs() / 60
)
    # keep all abx given at the chosen administration time (+- 10 mins) for that infection

    tolerance_minutes = 10


    complete_prophylaxis = merged[
        merged["time_diff_minutes"] <= tolerance_minutes
    ].copy()
    
    # if any non-therapeutic drugs exist at the chosen time for an infection_id,
    # keep only those; otherwise keep the therapeutic ones
    complete_prophylaxis = (
        complete_prophylaxis.groupby(idx_col, group_keys=False)
        .apply(lambda g: g[g["is_therapeutic"] == False] if (~g["is_therapeutic"]).any() else g)
        .copy()
    )

  
    # combine meds given together
    prophylaxis_summary = (
        complete_prophylaxis.groupby(idx_col)
        .agg(
            prophylaxis=(abx_col, lambda s: " | ".join(sorted(set(s.dropna())))),
            n_prophylaxis_agents=(abx_col, lambda s: len(set(s.dropna()))),
        )
        .reset_index()
    )

    result = all_ids.merge(first_candidate_row, on=idx_col, how="left")
    result = result.merge(prophylaxis_summary, on=idx_col, how="left")

    return result



def map_prophylaxis_class(df, abx_cfg = abx_cfg):

    
    df["prophylaxis_class"] = df["prophylaxis"].map(abx_cfg['prophylaxis_class_mapping'])

    return df

def run(cfg_path, out_csv_name, save=True, return_df=False, verbose=False):

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    cfg_path = Path(cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})
    config_cols = cfg.get('columns', {})

    administrations_df = pd.read_csv(PROJECT_ROOT / paths['administrations'])
    mcs_df = pd.read_csv(PROJECT_ROOT / paths['clean_mcs'])

    mcs_df = get_abx_exposure(
        mcs_df,
        cfg_path=cfg_path,
        verbose=False
    )

    if save:
        out_path = INTERIM_DIR / out_csv_name
        mcs_df.to_csv(out_path, index=False)

    if return_df:
        return mcs_df
        

def regroup_sparse_prophylaxis_categories(
    df,
    prophylaxis_col="prophylaxis",
    outcome_col="esbl_status",
    output_col="prophylaxis_group",
    min_total=20,
    min_class_count=10,
    keep_categories=None
):
    """
    Collapse sparse prophylaxis categories into broader antibiotic groups
    for more stable modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe.

    prophylaxis_col : str
        Column containing prophylaxis regimens.

    outcome_col : str
        Binary outcome column (e.g. ESBL vs non-ESBL).

    output_col : str
        Name of grouped prophylaxis column.

    min_total : int
        Minimum total observations required to keep a category.

    min_class_count : int
        Minimum observations required in EACH outcome class.

    keep_categories : list or None
        Categories to never collapse into "other".

    Returns
    -------
    pd.DataFrame
        DataFrame with grouped prophylaxis column added.
    """

    df = df.copy()

    if keep_categories is None:
        keep_categories = ["no_prophylaxis"]

    # ---------------------------------------------------------
    # Helper function to regroup rare antibiotic regimens
    # ---------------------------------------------------------

    def group_abx(x):

        x = str(x).lower()

        if "co-amoxiclav" in x:
            return "co-amoxiclav"

        elif "cefuroxime" in x:
            return "cefuroxime"

        elif "ceftriaxone" in x:
            return "third_gen_ceph"

        elif "meropenem" in x:
            return "carbapenem"

        elif "ciprofloxacin" in x:
            return "fluoroquinolone"

        elif "vancomycin" in x or "teicoplanin" in x:
            return "glycopeptide_based"

        elif "clindamycin" in x:
            return "clindamycin"

        else:
            return "other"

    # ---------------------------------------------------------
    # FIRST PASS:
    # Find sparse ORIGINAL prophylaxis categories
    # ---------------------------------------------------------

    ct = pd.crosstab(
        df[prophylaxis_col],
        df[outcome_col]
    )

    ct = ct.reindex(
        columns=["non-ESBL", "ESBL"],
        fill_value=0
    )

    ct["total"] = ct["non-ESBL"] + ct["ESBL"]

    sparse_categories = ct[
        (ct["total"] < min_total) |
        (ct["non-ESBL"] < min_class_count) |
        (ct["ESBL"] < min_class_count)
    ].index

    # initialise grouped column
    df[output_col] = df[prophylaxis_col]

    # regroup sparse categories
    sparse_mask = df[prophylaxis_col].isin(sparse_categories)

    df.loc[sparse_mask, output_col] = (
        df.loc[sparse_mask, prophylaxis_col]
        .apply(group_abx)
    )

    # ---------------------------------------------------------
    # SECOND PASS:
    # Collapse still-sparse grouped categories into "other"
    # ---------------------------------------------------------

    ct_grouped = pd.crosstab(
        df[output_col],
        df[outcome_col]
    )

    ct_grouped = ct_grouped.reindex(
        columns=["non-ESBL", "ESBL"],
        fill_value=0
    )

    ct_grouped["total"] = (
        ct_grouped["non-ESBL"] +
        ct_grouped["ESBL"]
    )

    still_sparse = ct_grouped[
        (ct_grouped["total"] < min_total) |
        (ct_grouped["non-ESBL"] < min_class_count) |
        (ct_grouped["ESBL"] < min_class_count)
    ].index

    # keep clinically important categories
    still_sparse = [
        x for x in still_sparse
        if x not in keep_categories
    ]

    df.loc[
        df[output_col].isin(still_sparse),
        output_col
    ] = "other"


    prophylaxis_mapping = {

    # Aminoglycoside-based (without cephalosporin)
    "gentamicin": "aminoglycoside_based",
    "gentamicin | metronidazole": "aminoglycoside_based",
    "gentamicin | teicoplanin": "aminoglycoside_based",

    # Glycopeptide
    "vancomycin": "glycopeptide_based",

    "cefuroxime | gentamicin": "cephalosporin_aminoglycoside",
    "cefuroxime | gentamicin | metronidazole": "cephalosporin_aminoglycoside",}

    df[output_col] = df[output_col].replace(prophylaxis_mapping)

    return df



def get_abx_exposure(
    mcs_df: pd.DataFrame,
    cfg_path,
    verbose: bool = False,
) -> pd.DataFrame:
    
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})
    config_cols = cfg.get('columns', {})

    logging.info('Loading datasets...')

    administrations_path = PROJECT_ROOT / paths['abx_administrations']
    administrations_df = pd.read_csv(administrations_path)

    # --- filter abx ---
    administrations_df = dct.clean_df_columns(administrations_df)
    abx_administrations = filter_antibiotics(administrations_df)

    # --- merge ---
    patient_administrations = pd.merge(
        abx_administrations[['subject', 'medication_name_short','administration_dt', 'route']],
        mcs_df,
        how='right',
        on='subject'
    )

    # --- prophylaxis ---
    prophylaxis_df = infer_prophylaxis(
        patient_administrations,
        therapeutic_agents=abx_cfg['THERAPEUTIC_AGENTS'],
        idx_col="infection_id",  
        abx_col="medication_name_short",
        route_col="route",
    )


    mcs_df = mcs_df.merge(prophylaxis_df, on="infection_id", how="left")

    mcs_df = regroup_sparse_prophylaxis_categories(mcs_df)

    mcs_df = map_prophylaxis_class(mcs_df)
    
    # --- past abx ---
    past_abx_df = get_past_abx(
        patient_administrations,
    )

    mcs_df = mcs_df.merge(
        past_abx_df[["infection_id", "past_abx"]].drop_duplicates(),
        on="infection_id",
        how="left",
    )

    return mcs_df

    

if __name__ == '__main__':
    import sys 

    if len(sys.argv) != 3:
        logging.warning("Usage: python get_prophylaxis_past_abx.py <config.yaml> <mcs_output.csv>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    out_csv_name = sys.argv[2]

    run(cfg_path, out_csv_name)