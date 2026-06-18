"""Clean multivariable logistic regression results for manuscript output."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


MISSING_VALUES = ["", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "None", "none"]

OUTPUTS_DIR = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "first_episodes": OUTPUTS_DIR / "multivariable_lr" / "first_episodes" / "multivariable_lr_results.csv",
    "clustered_by_patient": OUTPUTS_DIR / "multivariable_lr" / "clustered_by_patient" / "multivariable_lr_results.csv",
}

OUTPUT_FILES = {
    "first_episodes": OUTPUTS_DIR / "multivariable_lr" / "first_episodes" / "multivariable_lr_results_cleaned.csv",
    "clustered_by_patient": OUTPUTS_DIR / "multivariable_lr" / "clustered_by_patient" / "multivariable_lr_results_cleaned.csv",
}


PREDICTOR_MAPPING = {
    "age_at_admission": "Age at admission",
    "gender": "Male sex",
    "hospital_exposure_90d": "Hospital exposure within 90 days",
    "past_abx": "Prior antibiotic exposure",
    "prior_positive_esbl": "Prior ESBL-positive culture",
    "surgery_length": "Surgery duration",
    "temp": "Temperature",
    "type2_diabetes": "Type 2 diabetes",
    "ethnicity_desc": "Ethnicity",
    "organism_bug": "Organism",
    "site": "Sample site",
    "tfc": "Surgical specialty",
    "prophylaxis_group": "Prophylaxis regimen",
}


CATEGORY_MAPPING = {
    "asian": "Asian",
    "black": "Black",
    "white": "White",
    "escherichia coli": "E. coli",
    "klebsiella oxytoca": "K. oxytoca",
    "klebsiella pneumoniae": "K. pneumoniae",
    "proteus mirabilis": "P. mirabilis",
    "blood": "Blood",
    "drain": "Drain",
    "sputum": "Sputum",
    "tips_devices": "Tips/devices",
    "tissue/biopsy": "Tissue/biopsy",
    "urine": "Urine",
    "wound": "Wound",
    "general_surgery": "General surgery",
    "abdominal_gi": "Abdominal gastrointestinal surgery",
    "cardiothoracic": "Cardiothoracic surgery",
    "gynaecological_oncology": "Gynaecological oncology",
    "neuro_ent": "Neurology/ENT",
    "obs_gynae": "Obstetrics/Gynaecology",
    "ortho_plastics": "Orthopaedics/Plastics",
    "uro_nephro": "Urology/Nephrology",
    "vascular": "Vascular surgery",
    "cefuroxime +/- metronidazole": "Cefuroxime ± metronidazole",
    "aminoglycoside-based": "Aminoglycoside-based",
    "cefuroxime +/- metronidazole + aminoglycoside": "Cefuroxime ± metronidazole + aminoglycoside",
    "co-amoxiclav-based": "Co-amoxiclav-based",
    "limited_enterobact_activity": "Limited Enterobacterales activity",
    "no_prophylaxis": "No prophylaxis",
    "other_broad_spectrum_gram_negative_active": "Other broad-spectrum Gram-negative active",
}


REFERENCE_MAPPING = {
    "ethnicity_desc": "White",
    "organism_bug": "E. coli",
    "site": "Blood",
    "tfc": "General surgery",
    "prophylaxis_group": "Cefuroxime ± metronidazole",
}


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    for old, new in [("/", "_"), ("-", "_"), (" ", "_")]:
        name = name.replace(old, new)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def nice_label_from_raw(value: str) -> str:
    if pd.isna(value):
        return pd.NA

    label = str(value).strip().replace("_", " ").title()

    replacements = {
        "Copd": "COPD",
        "Crp": "CRP",
        "Esbl": "ESBL",
        "Imd": "IMD",
        "Tfc": "TFC",
        "Abx": "antibiotic",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    return label


def format_p_value(p):
    if pd.isna(p):
        return pd.NA

    p = float(p)

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [normalize_column_name(col) for col in df.columns]

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(MISSING_VALUES, pd.NA)

    numeric_cols = ["odds_ratio", "ci_lower", "ci_upper", "p_value", "n", "events"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Variable"] = df["predictor"].map(PREDICTOR_MAPPING)
    df["Variable"] = df["Variable"].fillna(df["predictor"].apply(nice_label_from_raw))

    df["Subcategory"] = ""

    categorical_mask = df["is_categorical"].eq(True)
    continuous_mask = df["is_categorical"].eq(False)

    df.loc[categorical_mask, "Subcategory"] = (
        df.loc[categorical_mask, "category"]
        .replace(CATEGORY_MAPPING)
        .fillna("")
    )

    df.loc[continuous_mask, "Subcategory"] = "Per unit increase"

    df.loc[
        continuous_mask & df["predictor"].isin([
            "gender",
            "hospital_exposure_90d",
            "past_abx",
            "prior_positive_esbl",
            "type2_diabetes",
        ]),
        "Subcategory",
    ] = "Yes"

    df["Reference category"] = ""

    df.loc[categorical_mask, "Reference category"] = (
        df.loc[categorical_mask, "predictor"]
        .replace(REFERENCE_MAPPING)
        .fillna("")
    )

    df.loc[continuous_mask, "Reference category"] = "No"

    df.loc[
        continuous_mask & df["predictor"].isin([
            "age_at_admission",
            "surgery_length",
            "temp",
        ]),
        "Reference category",
    ] = "Per unit increase"

    df["Odds ratio (95% CI)"] = (
        df["odds_ratio"].round(2).map(lambda x: f"{x:.2f}")
        + " ("
        + df["ci_lower"].round(2).map(lambda x: f"{x:.2f}")
        + "–"
        + df["ci_upper"].round(2).map(lambda x: f"{x:.2f}")
        + ")"
    )

    df["P-value"] = df["p_value"].apply(format_p_value)

    final_df = df[
        [
            "Variable",
            "Subcategory",
            "Reference category",
            "Odds ratio (95% CI)",
            "P-value",
            "n",
            "events",
        ]
    ].copy()

    return final_df


def clean_one_file(input_csv: Path, output_csv: Path) -> None:
    raw_df = pd.read_csv(input_csv)
    cleaned_df = clean_dataframe(raw_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_csv, index=False)

    print(f"Saved cleaned results to: {output_csv}")


def main() -> None:
    for name in INPUT_FILES:
        clean_one_file(INPUT_FILES[name], OUTPUT_FILES[name])


if __name__ == "__main__":
    main()