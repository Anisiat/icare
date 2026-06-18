"""Clean univariable logistic regression results for manuscript output."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


MISSING_VALUES = ["", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "None", "none"]

OUTPUTS_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = (
    OUTPUTS_DIR / "univariable_lr" / "first_episodes" / "univariable_lr_results.csv"
)

DEFAULT_OUTPUT_CSV = (
    OUTPUTS_DIR / "univariable_lr" / "first_episodes" / "univariable_lr_results_cleaned.csv"
)


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

    "n_surgeries": "Number of surgeries",
    "surgery_length": "Surgery duration",
    "min_days_from_surgery_to_infection": "Days from surgery to infection",
    "prophylaxis_group": "Prophylaxis regimen",

    "site": "Infection site",
    "n_sites": "Number of infection sites",
    "organism_bug": "Organism",
    "n_samples": "Number of microbiology samples",
    "culture_span_days": "Culture span",

    "temp": "Temperature",
    "crp": "CRP",

    "length_of_stay_days": "Length of stay",
    "in_hosp_mortality": "In-hospital mortality",
}


CATEGORY_MAPPING = {
    "asian": "Asian",
    "black": "Black",
    "white": "White",
    "mixed": "Mixed",
    "other": "Other",
    "unknown": "Unknown",

    "escherichia coli": "E. coli",
    "proteus mirabilis": "P. mirabilis",
    "klebsiella oxytoca": "K. oxytoca",
    "klebsiella pneumoniae": "K. pneumoniae",

    "cefuroxime +/- metronidazole": "Cefuroxime ± metronidazole",
    "co-amoxiclav-based": "Co-amoxiclav-based",
    "aminoglycoside-based": "Aminoglycoside-based",
    "limited_enterobact_activity": "Limited Enterobacterales activity",
    "no_prophylaxis": "No prophylaxis",
    "other_broad_spectrum_gram_negative_active": (
        "Other broad-spectrum Gram-negative active"
    ),
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

    "Number of surgeries",
    "Surgery duration",
    "Days from surgery to infection",
    "Prophylaxis regimen",

    "Infection site",
    "Number of infection sites",
    "Organism",
    "Number of microbiology samples",
    "Culture span",

    "Temperature",
    "CRP",

    "Length of stay",
    "In-hospital mortality",
]


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    for old, new in [("/", "_"), ("-", "_"), (" ", "_")]:
        name = name.replace(old, new)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def nice_label_from_raw(value: str) -> str:
    """Fallback label if a predictor/category is not in the mapping."""
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().replace("_", " ")
    label = value.title()

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

    # Clean column names
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Clean string values
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(MISSING_VALUES, pd.NA)

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Convert numeric columns
    numeric_cols = ["odds_ratio", "ci_lower", "ci_upper", "p_value", "n", "events"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Publication-ready variable names
    df["Variable"] = df["predictor"].map(PREDICTOR_MAPPING)
    df["Variable"] = df["Variable"].fillna(df["predictor"].apply(nice_label_from_raw))

    # Publication-ready category/reference names
    for col in ["category", "reference"]:
        if col in df.columns:
            df[col] = df[col].replace(CATEGORY_MAPPING)
            df[col] = df[col].apply(
                lambda x: nice_label_from_raw(x) if pd.notna(x) else pd.NA
            )

    # Binary variables
    df["Comparison"] = "Yes vs No"

    categorical_mask = df["model_type"].eq("categorical")
    continuous_mask = df["model_type"].eq("continuous")

    df.loc[continuous_mask, "Comparison"] = "Per unit increase"

    df.loc[categorical_mask, "Comparison"] = (
        df.loc[categorical_mask, "category"].astype("string")
        + " vs "
        + df.loc[categorical_mask, "reference"].astype("string")
    )

    # OR with CI
    df["Odds ratio (95% CI)"] = (
        df["odds_ratio"].round(2).map(lambda x: f"{x:.2f}")
        + " ("
        + df["ci_lower"].round(2).map(lambda x: f"{x:.2f}")
        + "–"
        + df["ci_upper"].round(2).map(lambda x: f"{x:.2f}")
        + ")"
    )

    df["P-value"] = df["p_value"].apply(format_p_value)

    # Reorder variables
    df["variable_order"] = df["Variable"].apply(
        lambda x: VARIABLE_ORDER.index(x) if x in VARIABLE_ORDER else 999
    )

    df = df.sort_values(["variable_order", "Variable", "Comparison"]).reset_index(drop=True)

    final_df = df[
        [
            "Variable",
            "Comparison",
            "Odds ratio (95% CI)",
            "P-value",
        ]
    ].copy()

    # Only blank repeated variable names AFTER sorting.
    # This keeps the first row visible.
    repeated_mask = final_df["Variable"].eq(final_df["Variable"].shift())
    final_df.loc[repeated_mask, "Variable"] = ""

    return final_df


def import_raw_csv(file_path: Path = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    return pd.read_csv(file_path)


def main() -> None:
    raw_df = import_raw_csv()
    cleaned_df = clean_dataframe(raw_df)

    cleaned_df.to_csv(DEFAULT_OUTPUT_CSV, index=False)

    print(f"Saved cleaned results to: {DEFAULT_OUTPUT_CSV}")
    print(cleaned_df.head(30))


if __name__ == "__main__":
    main()