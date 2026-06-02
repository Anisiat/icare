import numpy as np
import pandas as pd
from typing import List


def infer_prophylaxis(
    df: pd.DataFrame,
    therapeutic_agents: List[str],
    surgery_start_col: str = "surgery_start_dt",
    administration_dt_col: str = "administration_dt",
    idx_col: str = "infection_id",
    abx_col: str = "medication_name_short",
    rout_dfe_col: str = "rout_dfe",
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
        rout_dfe_col,
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
    out_df[rout_dfe_col] = out_df[rout_dfe_col].astype(str).str.strip().str.lower()
    out_df[abx_col] = out_df[abx_col].astype(str).str.strip().str.lower()

    # IV only
    out_df = out_df[out_df[rout_dfe_col] == "iv"].copy()

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
    out_df["tier"] = np.where(pass1_mask, 1, np.where(pass2_mask, 2, np.nan))

    # drop rows where administration is in neither time windows
    # keeps 
    candidates = out_df.dropna(subset=["tier"]).copy()

    all_ids = df[[idx_col]].drop_duplicates()

    if candidates.empty:
        return all_ids.assign(
            prophylaxis=pd.NA,
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

    # combine meds given together
    prophylaxis_summary = (
        complete_prophylaxis.groupby(idx_col)
        .agg(
            prophylaxis=(abx_col, lambda s: "|".join(sorted(set(s.dropna())))),
            n_prophylaxis_agents=(abx_col, lambda s: len(set(s.dropna()))),
        )
        .reset_index()
    )

    result = all_ids.merge(first_candidate_row, on=idx_col, how="left")
    result = result.merge(prophylaxis_summary, on=idx_col, how="left")

    return result