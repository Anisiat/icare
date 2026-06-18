"""Clean univariable and multivariable logistic regression results for manuscript output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MISSING_VALUES = ["", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "None", "none"]

OUTPUTS_DIR = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "univariable_first_episodes": (
        OUTPUTS_DIR / "univariable_lr" / "first_episodes" / "univariable_lr_results.csv"
    ),
    "multivariable_first_episodes": (
        OUTPUTS_DIR / "multivariable_lr" / "first_episodes" / "multivariable_lr_results.csv"
    ),
    "multivariable_clustered_patient": (
        OUTPUTS_DIR / "multivariable_lr" / "clustered_by_patient" / "multivariable_lr_results.csv"
    ),
}

OUTPUT_FILES = {
    "univariable_first_episodes": (
        OUTPUTS_DIR / "univariable_lr" / "first_episodes" / "univariable_lr_results_cleaned.csv"
    ),
    "multivariable_first_episodes": (
        OUTPUTS_DIR / "multivariable_lr" / "first_episodes" / "multivariable_lr_results_cleaned.csv"
    ),
    "multivariable_clustered_patient": (
        OUTPUTS_DIR / "multivariable_lr" / "clustered_by_patient" / "multivariable_lr_results_cleaned.csv"
    ),
}


PREDICTOR_MAPPING = {
    "age_at_admission": "Age at admission",
    "gender": "Male sex",
    "ethnicity_desc": "Ethnicity",
    "imd_decile": "Index of Multiple Deprivation decile",
    "asthma": "Asthma",
    "cancer": "Cancer",
    "copd": "COPD",
    "hypertension": "Hypertension",
    "ischaemic_heart_disease": "Ischaemic heart disease",
    "type2_diabetes": "Type 2 diabetes",
    "emergency_exposure_90d": "Emergency exposure within 90 days",
    "hospital_exposure_90d": "Hospital exposure within 90 days",
    "any_healthcare_exposure_90d": "Any healthcare exposure within 90 days",
    "past_abx": "Prior antibiotic exposure",
    "prior_positive_esbl": "Prior ESBL-positive culture",
    "tfc": "Surgical specialty",
    "surgery_length": "Surgery duration",
    "min_days_from_surgery_to_infection": "Days from surgery to infection",
    "days_from_admission_to_first_culture": "Days from admission to first culture",
    "prophylaxis_group": "Prophylaxis regimen",
    "site": "Sample site",
    "organism_bug": "Organism",
    "temp": "Temperature",
    "crp": "CRP",
}


CATEGORY_MAPPING = {
    "white": "White",
    "asian": "Asian",
    "black": "Black",
    "mixed": "Mixed",
    "other": "Other",
    "escherichia coli": "E. coli",
    "proteus mirabilis": "P. mirabilis",
    "klebsiella oxytoca": "K. oxytoca",
    "klebsiella pneumoniae": "K. pneumoniae",
    "cefuroxime +/- metronidazole": "Cefuroxime ± metronidazole",
    "cefuroxime +/- metronidazole + aminoglycoside": (
        "Cefuroxime ± metronidazole + aminoglycoside"
    ),
    "co-amoxiclav-based": "Co-amoxiclav-based",
    "aminoglycoside-based": "Aminoglycoside-based",
    "limited_enterobact_activity": "Limited Enterobacterales activity",
    "no_prophylaxis": "No prophylaxis",
    "other_broad_spectrum_gram_negative_active": (
        "Other broad-spectrum Gram-negative active"
    ),
    "general_surgery": "General surgery",
    "abdominal_gi": "Abdominal gastrointestinal surgery",
    "cardiothoracic": "Cardiothoracic surgery",
    "gynaecological_oncology": "Gynaecological oncology",
    "neuro_ent": "Neurology/ENT",
    "obs_gynae": "Obstetrics/Gynaecology",
    "ortho_plastics": "Orthopaedics/Plastics",
    "uro_nephro": "Urology/Nephrology",
    "vascular": "Vascular surgery",
    "blood": "Blood",
    "drain": "Drain",
    "sputum": "Sputum",
    "tips_devices": "Tips/devices",
    "tissue/biopsy": "Tissue/biopsy",
    "urine": "Urine",
    "wound": "Wound",
}


VARIABLE_ORDER = [
    "Age at admission",
    "Male sex",
    "Ethnicity",
    "Index of Multiple Deprivation decile",
    "Asthma",
    "Cancer",
    "COPD",
    "Hypertension",
    "Ischaemic heart disease",
    "Type 2 diabetes",
    "Emergency exposure within 90 days",
    "Hospital exposure within 90 days",
    "Any healthcare exposure within 90 days",
    "Prior antibiotic exposure",
    "Prior ESBL-positive culture",
    "Surgical specialty",
    "Surgery duration",
    "Days from surgery to infection",
    "Days from admission to first culture",
    "Prophylaxis regimen",
    "Sample site",
    "Organism",
    "Temperature",
    "CRP",
]


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()

    for old, new in [("/", "_"), ("-", "_"), (" ", "_")]:
        name = name.replace(old, new)

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


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

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    numeric_cols = ["odds_ratio", "ci_lower", "ci_upper", "p_value", "n", "events"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "predictor" not in df.columns:
        raise ValueError(f"No predictor column found. Columns are: {df.columns.tolist()}")
    

    df["Variable"] = df["predictor"].replace(PREDICTOR_MAPPING)

    if "category" in df.columns:
        df["category"] = df["category"].replace(CATEGORY_MAPPING)

    if "reference" in df.columns:
        df["reference"] = df["reference"].replace(CATEGORY_MAPPING)

    df["Comparison"] = "Yes vs No"

    if "model_type" in df.columns:
        categorical_mask = df["model_type"].eq("categorical")
        continuous_mask = df["model_type"].eq("continuous")

        df.loc[continuous_mask, "Comparison"] = "Per unit increase"

        df.loc[categorical_mask, "Comparison"] = (
            df.loc[categorical_mask, "category"].astype("string")
            + " vs "
            + df.loc[categorical_mask, "reference"].astype("string")
        )

    df["Odds ratio (95% CI)"] = (
        df["odds_ratio"].round(2).map(lambda x: f"{x:.2f}")
        + " ("
        + df["ci_lower"].round(2).map(lambda x: f"{x:.2f}")
        + "–"
        + df["ci_upper"].round(2).map(lambda x: f"{x:.2f}")
        + ")"
    )

    df["P-value"] = df["p_value"].apply(format_p_value)

    df["variable_order"] = df["Variable"].apply(
        lambda x: VARIABLE_ORDER.index(x) if x in VARIABLE_ORDER else 999
    )

    df = df.sort_values(["variable_order", "Variable", "Comparison"])

    final_df = df[
        [
            "Variable",
            "Comparison",
            "Odds ratio (95% CI)",
            "P-value",
        ]
    ].copy()
    
    final_df = final_df.drop_duplicates().reset_index(drop=True)

    full_variables = final_df["Variable"].copy()

    repeated_mask = final_df["Variable"].eq(final_df["Variable"].shift())
    final_df.loc[repeated_mask, "Variable"] = ""

    final_df.loc[0, "Variable"] = full_variables.iloc[0]

    return final_df


def clean_one_file(input_path: Path, output_path: Path) -> None:
    print(f"\nCleaning: {input_path}")

    raw_df = pd.read_csv(input_path)
    cleaned_df = clean_dataframe(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Saved cleaned file to: {output_path}")
    print(cleaned_df.head(20))


def main() -> None:
    jobs = [
        "univariable_first_episodes",
        "multivariable_first_episodes",
        "multivariable_clustered_patient",
    ]

    for name in jobs:
        clean_one_file(
            INPUT_FILES[name],
            OUTPUT_FILES[name],
        )


if __name__ == "__main__":
    main()