import pandas as pd
from pathlib import Path



def infer_prophylaxis(
    df: pd.DataFrame,
    config_cols: dict,
    surgery_start_col: str = "surgery_start_dt",
    administration_dt_col: str = "administration_dt",
    lab_col: str = "lab_test_id",
    med_col: str = "medication_name_short",
    route_col: str = "route",
    hours_before_surgery: int = 12,
    hours_after_surgery: int = 1,
) -> pd.DataFrame:
    """
    For each lab_test_id, infer a single prophylaxis agent.

    Rules:
    - Consider only IV administrations in the window
      [-hours_after_surgery, +hours_before_surgery] around surgery start.
    - Prefer prophylactic agents (not in the therapeutic list).
    - Among prophylactic agents, choose the one closest to incision
      (smallest absolute delta_hours_prophylaxis).
    - If no prophylactic agent exists for a lab_test_id, choose the
      therapeutic agent closest to incision.
    - If no administration in the window at all, label as 'no_recorded_prophylaxis'.
    """

    required = [surgery_start_col, administration_dt_col, lab_col, med_col, route_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # ---- 1. convert times ----
    for col in config_cols["mcs_time_cols"]:
        df[col] = pd.to_datetime(df[col])

    # ---- 2. compute delta (positive = before surgery) ----
    df = df.copy()
    df["delta_hours_prophylaxis"] = (
        df[surgery_start_col] - df[administration_dt_col]
    ).dt.total_seconds() / 3600

    # ---- 3. IV only ----
    iv_df = df[df[route_col].str.lower() == "iv"].copy()

    # ---- 4. time window filter ----
    window_mask = (
        (iv_df["delta_hours_prophylaxis"] >= -hours_after_surgery)
        & (iv_df["delta_hours_prophylaxis"] <= hours_before_surgery)
    )
    candidates = iv_df.loc[window_mask].copy()

    # if no candidates at all, everything is "no_recorded_prophylaxis"
    if candidates.empty:
        out = (
            df[[lab_col]]
            .drop_duplicates()
            .assign(
                prophylaxis="no_recorded_prophylaxis",
                delta_hours_prophylaxis=pd.NA,
            )
        )
        return out

    # ---- 5. mark therapeutic vs prophylactic ----
    therapeutic_agents = [
        "piperacillin-tazobactam",
        "ciprofloxacin",
        "meropenem",
        "amikacin",
        "ceffriaxone",
        "colistin",
        "linezolid",
        "avibactam-ceftazidime",
        "ceftazidime",
        "fosfomycin",
        "cefiderocol",
        "amoxicillin (contains penicillin)",
    ]
    therapeutic_set = {a.lower() for a in therapeutic_agents}

    candidates["is_therapeutic"] = (
        candidates[med_col].str.lower().isin(therapeutic_set)
    )

    # ---- 6. distance to incision ----
    candidates["abs_delta"] = candidates["delta_hours_prophylaxis"].abs()

    # ---- 7. sort by (lab, is_therapeutic, abs_delta) ----
    # this implements:
    #   - prophylactic (False) before therapeutic (True)
    #   - then closest to incision (smallest abs_delta)
    sorted_cand = candidates.sort_values(
        [lab_col, "is_therapeutic", "abs_delta"],
        ascending=[True, True, True],
    )

    # ---- 8. pick the first row per lab_test_id ----
    best_rows = (
        sorted_cand.groupby(lab_col, as_index=False)
        .first()
    )

    best_proph = best_rows[
        [lab_col, med_col, "delta_hours_prophylaxis", "is_therapeutic"]
    ].rename(columns={med_col: "prophylaxis"})

    # ---- 9. merge onto unique lab ids; fill missing ----
    # Start from all labs present in the original df
    labs = df[[lab_col]].drop_duplicates()

    out = labs.merge(best_proph, on=lab_col, how="left")

    out["prophylaxis"] = out["prophylaxis"].fillna("no_recorded_prophylaxis")

    # returns columns: lab_test_id, prophylaxis, "delta_hours_prophylaxis", is_therapeutic

    return out

import pandas as pd
from typing import Iterable, List


def resolve_no_recorded_prophylaxis(
    patient_admin_df: pd.DataFrame,
    no_prophylaxis_ids: Iterable,
    therapeutic_agents: List[str],
    hours_window: float = 12.0,
    lab_col: str = "lab_test_id",
    med_col: str = "medication_name_short",
    route_col: str = "route",
    admin_dt_col: str = "administration_dt",
    admit_date_col: str = "admission_date",
    discharge_date_col: str = "discharge_date",
    delta_col: str = "delta_hours_prophylaxis",
) -> pd.DataFrame:
    """
    For lab_test_ids currently labelled 'no_recorded_prophylaxis',
    look across the whole admission for IV antibiotic administrations
    and try to infer a likely prophylaxis agent.

    Logic:
    - Restrict to rows where:
        * lab_test_id in no_prophylaxis_ids
        * administration_dt between admission_date and discharge_date
        * route == 'iv'
        * |delta_hours_prophylaxis| < hours_window
    - Mark each administration as therapeutic/non-therapeutic using
      `therapeutic_agents` list.
    - For each lab_test_id:
        * Prefer a non-therapeutic agent (prophylactic) closest to incision
          (min |delta|).
        * If none exist, fall back to the therapeutic agent closest to incision.
        * If no administrations satisfy the filters, keep 'no_recorded_prophylaxis'.

    Returns a DataFrame with one row per lab_test_id:
        [lab_test_id, resolved_prophylaxis, resolved_delta_hours_prophylaxis,
         resolved_is_therapeutic]
    """

    # Start from only labs we want to "rescue"
    df = patient_admin_df[
        patient_admin_df[lab_col].isin(no_prophylaxis_ids)
    ].copy()

    if df.empty:
        # nothing to resolve, return an empty mapping
        return pd.DataFrame(
            columns=[
                lab_col,
                "resolved_prophylaxis",
                "resolved_delta_hours_prophylaxis",
                "resolved_is_therapeutic",
            ]
        )

    # Ensure datetimes
    df[admin_dt_col] = pd.to_datetime(df[admin_dt_col])
    df[admit_date_col] = pd.to_datetime(df[admit_date_col])
    df[discharge_date_col] = pd.to_datetime(df[discharge_date_col])

    # 1) Keep administrations during admission
    mask_adm = df[admin_dt_col].dt.date.between(
        df[admit_date_col].dt.date,
        df[discharge_date_col].dt.date,
    )
    df = df[mask_adm].copy()

    # 2) IV only
    df = df[df[route_col].str.lower() == "iv"].copy()

    # 3) Within +/- hours_window of surgery
    df = df[df[delta_col].abs() < hours_window].copy()

    if df.empty:
        # still nothing → all remain 'no_recorded_prophylaxis'
        out = (
            pd.DataFrame({lab_col: list(no_prophylaxis_ids)})
            .drop_duplicates()
            .assign(
                resolved_prophylaxis="no_recorded_prophylaxis",
                resolved_delta_hours_prophylaxis=pd.NA,
                resolved_is_therapeutic=pd.NA,
            )
        )
        return out


    

    # 4) Mark therapeutic vs prophylactic
    therapeutic_set = {a.lower() for a in therapeutic_agents}
    df["resolved_is_therapeutic"] = df[med_col].str.lower().isin(therapeutic_set)

    # 5) Distance to incision
    df["abs_delta"] = df[delta_col].abs()

    # 6) Sort: per lab -> prophylactic first, then closest in time
    df_sorted = df.sort_values(
        [lab_col, "resolved_is_therapeutic", "abs_delta"],
        ascending=[True, True, True],
    )

    # 7) Pick first row per lab_test_id
    best_rows = df_sorted.groupby(lab_col, as_index=False).first()

    # 8) Build output mapping
    out = best_rows[
        [lab_col, med_col, delta_col, "resolved_is_therapeutic"]
    ].rename(
        columns={
            med_col: "resolved_prophylaxis",
            delta_col: "resolved_delta_hours_prophylaxis",
        }
    )

    # 9) Some lab_ids may still have no rows after filters → fill as 'no_recorded'
    all_labs = pd.DataFrame({lab_col: list(no_prophylaxis_ids)}).drop_duplicates()
    out = all_labs.merge(out, on=lab_col, how="left")

    out["resolved_prophylaxis"] = out["resolved_prophylaxis"].fillna(
        "no_recorded_prophylaxis"
    )

    return out


       
def infer_prophylaxis(df,
    config_cols,
    therapeutic_agents,
    surgery_start_col = 'surgery_start_dt',
    administration_dt_col =  'administration_dt', 
    hours_after_surgery = 1,
    hours_before_surgery = 12
    ):

    required = [surgery_start_col, administration_dt_col, 'route']

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # converting time columns
    for col in config_cols['mcs_time_cols']:
        df[col] = pd.to_datetime(df[col])

    df['administration_dt'] = pd.to_datetime(df['administration_dt'])

        # filter for IV route only 
    df = df[df['route'] == 'iv']

    # Time between administration and surgery start (positive values)
    df['delta_hours_prophylaxis'] = (df[surgery_start_col] - 
                                                        df[administration_dt_col]).dt.total_seconds() / 3600

    # time window mask 
    mask_window = (
        (-hours_after_surgery <= df['delta_hours_prophylaxis'] ) & 
        (df['delta_hours_prophylaxis'] <= hours_before_surgery)
    )

    # get all administrations for the prophylaxis time window 
    prophylaxis_administrations = df.loc[mask_window].copy()

    # mark therapeutic agents 
    prophylaxis_administrations['is_therapeutic'] = prophylaxis_administrations['medication_name_short'].isin(therapeutic_agents)

    # distance to incision
    prophylaxis_administrations['delta_hours_prophylaxis'] = prophylaxis_administrations['delta_hours_prophylaxis'].abs()

    # sort values:
        # 1. prophylaxis before therapeutic
        # 2. closest to incision

    sorted_prophyhlaxis = (
        prophylaxis_administrations
        .sort_values(['lab_test_id', 'is_therapeutic', 'delta_hours_prophylaxis'],
         ascending= [True, True, True]))

    prophylaxis_df = sorted_prophyhlaxis.groupby('lab_test_id', as_index = False).first()
    
    out_df = (prophylaxis_df[['lab_test_id', 'medication_name_short', 'is_therapeutic', 'delta_hours_prophylaxis' ]]
    .rename(columns = {'medication_name_short' : 'prophylaxis'}))

    all_labs = df[['lab_test_id']].drop_duplicates()

    out_df = all_labs.merge(out_df, on = 'lab_test_id', how = 'left')

    out_df['prophylaxis'] = out_df['prophylaxis'].fillna('no_recorded_prophylaxis')
  
    return out_df



#### use case 

no_proph_ids = unique_samples.loc[
    unique_samples["prophylaxis"] == "no_recorded_prophylaxis", "lab_test_id"
].unique()

therapeutic_agents = [
    "piperacillin-tazobactam",
    "ciprofloxacin",
    "meropenem",
    "amikacin",
    "ceffriaxone",
    "colistin",
    "linezolid",
    "avibactam-ceftazidime",
    "ceftazidime",
    "fosfomycin",
    "cefiderocol",
    "amoxicillin (contains penicillin)",
]

resolved = resolve_no_recorded_prophylaxis(
    patient_admin_df=patient_administrations,
    no_prophylaxis_ids=no_proph_ids,
    therapeutic_agents=therapeutic_agents,
    hours_window=12,
)

# Join back to unique_samples and overwrite 'no_recorded_prophylaxis' where resolved
unique_samples = unique_samples.merge(
    resolved[[ "lab_test_id", "resolved_prophylaxis" ]],
    on="lab_test_id",
    how="left",
)

mask = unique_samples["prophylaxis"] == "no_recorded_prophylaxis"
unique_samples.loc[mask, "prophylaxis"] = unique_samples.loc[
    mask, "resolved_prophylaxis"
]
unique_samples = unique_samples.drop(columns=["resolved_prophylaxis"])




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