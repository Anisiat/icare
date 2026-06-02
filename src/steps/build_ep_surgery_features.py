import pandas as pd
import logging
from pathlib import Path
import sys

# === GET FILE PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import data_cleaning_tools as dct


# =========================
# CORE CLEANING FUNCTION
# =========================
def get_surgery_features(df: pd.DataFrame) -> pd.DataFrame:

    tfc_cfg = dct.load_config("../configs/tfc.yaml")

    
    
    meta_labels = {
        "emergency ncepod (surg)",
        "not in enumerations list",
        "other operative procedure (surg)",
    }

    df = df.copy()
    
    # -------------------------
    # 1. Normalise procedure text
    # -------------------------
    df["procedure_desc"] = df["procedure_desc"].astype(str).str.lower()

    # -------------------------
    # 2. Drop meta labels IF real procedure exists in group
    # -------------------------
    has_real_proc = (
        df.groupby("infection_id")["procedure_desc"]
        .transform(lambda s: (~s.isin(meta_labels)).any())
    )

    # only keep surgeries that are not in meta labvels or keep surgeries if they don't have a real procedure description 
    mask_keep = (~df["procedure_desc"].isin(meta_labels)) | (~has_real_proc)
    df = df[mask_keep].copy()

    # -------------------------
    # 3. Datetime handling
    # -------------------------
    df["surgery_start_dt"] = pd.to_datetime(df["surgery_start_dt"], errors="coerce")
    df["infection_ep_start"] = pd.to_datetime(df["infection_ep_start"], errors="coerce")

    # -------------------------
    # 4. Time delta: surgery → infection
    # -------------------------
    df["days_from_surgery_to_infection"] = (
        df["infection_ep_start"] - df["surgery_start_dt"]
    ).dt.days

    # -------------------------
    # 5. Surgery before infection flag
    # -------------------------
    df["surgery_before_infection"] = df["days_from_surgery_to_infection"] >= 0

    # -------------------------
    # 6. Emergency flag (from NCEPOD)
    # -------------------------
    df["is_emergency"] = df["procedure_desc"].str.contains("ncepod", na=False)

    # -------------------------
    # 7. Compute surgery length 
    # -------------------------
    
    df['surgery_length'] = df['surgery_stop_dt'] - df['surgery_start_dt']
    df['surgery_length_hours'] = (
        df['surgery_length'].dt.total_seconds() / 3600
    )
    
    df.drop('surgery_length', axis = 1, inplace = True)


    # -------------------------
    # 8. Map TFC
    # -------------------------

    df["tfc_desc"] = (
    df["tfc_desc"]
    .str.strip()
    .str.lower()
    )
    
    df["tfc_group"] = df["tfc_desc"].map(tfc_cfg['tfc_mapping'])


    return df


# =========================
# RUN WRAPPER
# =========================
def run(
    cfg_path,
    out_csv_name = 'mcs_surgery_clean.csv',
    save=True,
    return_df=False,
    verbose=False,
):

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    logging.info("Loading config file...")
    cfg_path = Path(cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get("paths", {})

    # -------------------------
    # Load input
    # -------------------------
    logging.info("Loading surgery dataset...")
    surg_path = PROJECT_ROOT / paths["clean_mcs"]
    df = pd.read_csv(surg_path)

    # -------------------------
    # Clean
    # -------------------------
    df_clean = get_surgery_features(df, cfg)

    # -------------------------
    # Save
    # -------------------------
    if save:
        out_path = PROJECT_ROOT / out_csv_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out_path, index=False)
        logging.info(f"Saved cleaned surgeries to {out_path}")

    if return_df:
        return df_clean