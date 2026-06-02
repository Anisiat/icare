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

cfg = dct.load_config("../configs/config.yaml")



def get_healthcare_exposure(
    df,
    cfg_paths,
    exposure_type="any",
    index_cols=("subject", "admission_date"),
    window_days=90,
):
    """
    Add binary healthcare exposure features in the window before admission.

    Emergency exposure:
      arrival_date / departure_date

    Hospital exposure:
      prior_admission_date / prior_discharge_date
    """

    df = df.copy()
    subject_col, admission_col = index_cols

    df[admission_col] = pd.to_datetime(df[admission_col], errors="coerce")

    def _load_exposure(path_key):
        exposure_path = PROJECT_ROOT / cfg_paths[path_key]
        exposure_raw = pd.read_csv(exposure_path)
        exposure_df = dct.clean_df_columns(exposure_raw)
        return exposure_df

    def _flag_prior_window(
        exposure_df,
        start_col,
        end_col,
        output_col,
    ):
        exposure_df = exposure_df.copy()

        exposure_df[start_col] = pd.to_datetime(exposure_df[start_col], errors="coerce")
        exposure_df[end_col] = pd.to_datetime(exposure_df[end_col], errors="coerce")

        tmp = df[[subject_col, admission_col]].drop_duplicates().copy()

        merged = tmp.merge(
            exposure_df[[subject_col, start_col, end_col]].drop_duplicates(),
            on=subject_col,
            how="left",
        )

        window_start = merged[admission_col] - pd.Timedelta(days=window_days)

        # exposure overlaps the 90d pre-admission window
        mask = (
            merged[start_col].notna()
            & merged[end_col].notna()
            & (merged[start_col] < merged[admission_col])
            & (merged[end_col] >= window_start)
        )

        flags = (
            merged.assign(**{output_col: mask.astype(int)})
            .groupby([subject_col, admission_col], as_index=False)[output_col]
            .max()
        )

        return flags

    exposure_flags = []

    if exposure_type in ["emergency", "any"]:
        emergency_df = _load_exposure("emergency_exposure")

        emergency_flags = _flag_prior_window(
            emergency_df,
            start_col="arrival_date",
            end_col="departure_date",
            output_col="emergency_exposure_90d",
        )

        exposure_flags.append(emergency_flags)

    if exposure_type in ["hospital", "any"]:
        hospital_df = _load_exposure("healthcare_exposure")

        hospital_flags = _flag_prior_window(
            hospital_df,
            start_col="prior_admission_date",
            end_col="prior_discharge_date",
            output_col="hospital_exposure_90d",
        )

        exposure_flags.append(hospital_flags)

    if exposure_type not in ["emergency", "hospital", "any"]:
        raise ValueError(
            "exposure_type must be one of: 'emergency', 'hospital', 'any'"
        )

    for flags in exposure_flags:
        df = df.merge(
            flags,
            on=[subject_col, admission_col],
            how="left",
        )

    for col in ["emergency_exposure_90d", "hospital_exposure_90d"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    if exposure_type == "any":
        df["any_healthcare_exposure_90d"] = (
            df[["emergency_exposure_90d", "hospital_exposure_90d"]]
            .max(axis=1)
            .astype(int)
        )

    return df
# def get_healthcare_exposure(
#     df,
#     cfg_paths,
#     exposure_type="emergency",
#     index_cols=("subject", "admission_date"),
# ):
#     """
#     Add binary healthcare exposure features to an analysis dataframe.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Main analysis dataframe.

#     cfg : dict
#         Project configuration dictionary.

#     exposure_type : str, default "emergency"
#         Type of exposure feature to create.

#         Options:
#         - "emergency"
#         - "hospital"
#         - "any"

#     index_cols : tuple or list of str
#         Columns used to match exposure records.

#     Returns
#     -------
#     pd.DataFrame
#         Analysis dataframe with added exposure feature(s).
#     """

#     df = df.copy()

#     def _load_exposure(path_key):

#         exposure_path = PROJECT_ROOT / cfg_paths[path_key]

#         exposure_raw = pd.read_csv(exposure_path)

#         exposure_df = dct.clean_df_columns(exposure_raw)

#         exposure_df = exposure_df.rename(
#             columns={"index_admission_date": "admission_date"}
#         )

#         return exposure_df[list(index_cols)].drop_duplicates()

#     if exposure_type == "emergency":

#         exposure_df = _load_exposure("emergency_exposure")

#         col_name = "emergency_exposure_90d"

#         exposed = exposure_df.assign(**{col_name: 1})

#     elif exposure_type == "hospital":

#         exposure_df = _load_exposure("healthcare_exposure")

#         col_name = "hospital_exposure_90d"

#         exposed = exposure_df.assign(**{col_name: 1})

#     elif exposure_type == "any":

#         emergency_df = _load_exposure("emergency_exposure")
#         hospital_df = _load_exposure("healthcare_exposure")

#         exposure_df = pd.concat(
#             [emergency_df, hospital_df],
#             ignore_index=True
#         ).drop_duplicates()

#         col_name = "any_healthcare_exposure_90d"

#         exposed = exposure_df.assign(**{col_name: 1})

#     else:
#         raise ValueError(
#             "exposure_type must be one of: "
#             "'emergency', 'hospital', 'any'"
#         )

#     df = df.merge(
#         exposed,
#         on=list(index_cols),
#         how="left"
#     )

#     df[col_name] = df[col_name].fillna(0).astype(int)

#     return df





# def get_healthcare_exposure(df, cfg, emergency = True, index_cols = ['subject', 'admission_date']):

    # df = df.copy()

    # paths = cfg.get("paths", {})

    # if emergency:
    #     emergency_path = PROJECT_ROOT / paths["emergency_exposure"]
    
    #     emergency_raw = pd.read_csv(emergency_path)
    #     emergency_df = dct.clean_df_columns(emergency_raw)
    
    #     exposure_df = emergency_df.rename(
    #         columns={"index_admission_date": "admission_date"}
    #     )
    #     col_name = 'emergency_exposure'
        
    # else:
    #     hospitalisation_path = PROJECT_ROOT / paths["healthcare_exposure"]

    #     healthcare_raw = pd.read_csv(emergency_path)
    #     healthcare_df = dct.clean_df_columns(emergency_raw)
    
    #     exposure_df = healthcare_df.rename(
    #         columns={"index_admission_date": "admission_date"}
    #     )
    #     col_name = 'hospital_exposure'

    # missing_df_cols = [col for col in index_cols if col not in df.columns]
    # missing_emergency_cols = [col for col in index_cols if col not in exposure_df.columns]

    # if missing_df_cols:
    #     raise ValueError(f"Missing columns in analysis df: {missing_df_cols}")

    # if missing_emergency_cols:
    #     raise ValueError(f"Missing columns in emergency df: {missing_emergency_cols}")

    # exposed = (
    #     exposure_df[list(index_cols)]
    #     .drop_duplicates()
    #     .assign(**{col_name: 1})
    # )

    # df = df.merge(
    #     exposed,
    #     on=list(index_cols),
    #     how="left"
    # )

    # df[col_name] = df[col_name].fillna(0).astype(int)

    # return df