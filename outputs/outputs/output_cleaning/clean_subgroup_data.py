"""Clean subgroup logistic regression results for manuscript output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MISSING_VALUES = ["", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "None", "none"]

OUTPUTS_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = OUTPUTS_DIR / "subgroup_analysis" / "subgroup_analysis_lr_results.csv"
OUTPUT_CSV = OUTPUTS_DIR / "subgroup_analysis" / "subgroup_analysis_lr_results_cleaned.csv"


TERM_MAPPING = {
    "gender": "Male sex",
    "hospital_exposure_90d": "Hospital exposure within 90 days",
    "past_abx": "Prior antibiotic exposure",
    "prior_positive_esbl": "Prior ESBL-positive culture",
    "surgery_length": "Surgery duration",
    "ethnicity_desc": "Ethnicity",
    "organism_bug": "Organism",
    "prophylaxis_group": "Prophylaxis regimen",
    "site": "Sample site",
    "tfc": "Surgical specialty",
}


LEVEL_MAPPING = {
    "abdominal_gi": "Abdominal gastrointestinal surgery",
    "cardiothoracic": "Cardiothoracic surgery",
    "general_surgery": "General surgery",
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
    "urine": "Urine",
    "wound": "Wound",

    "asian": "Asian",
    "black": "Black",
    "white": "White",

    "klebsiella pneumoniae": "K. pneumoniae",

    "aminoglycoside-based": "Aminoglycoside-based",
    "co-amoxiclav-based": "Co-amoxiclav-based",
    "limited_enterobact_activity": "Limited Enterobacterales activity",
    "no_prophylaxis": "No prophylaxis",
    "cefuroxime +/- metronidazole + aminoglycoside": (
        "Cefuroxime ± metronidazole + aminoglycoside"
    ),
    "other_broad_spectrum_gram_negative_active": (
        "Other broad-spectrum Gram-negative active"
    ),
}


def format_p_value(p):
    if pd.isna(p):
        return pd.NA

    p = float(p)

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def clean_term(term: str) -> tuple[str, str]:
    """
    Splits raw terms like:
    ethnicity_desc__asian
    into:
    Ethnicity, Asian vs reference
    """

    if pd.isna(term):
        return pd.NA, pd.NA

    term = str(term).strip()

    if "__" in term:
        variable, level = term.split("__", 1)
        variable_clean = TERM_MAPPING.get(variable, variable.replace("_", " ").title())
        level_clean = LEVEL_MAPPING.get(level, level.replace("_", " ").title())

        comparison = f"{level_clean} vs reference"
        return variable_clean, comparison

    variable_clean = TERM_MAPPING.get(term, term.replace("_", " ").title())

    if term == "surgery_length":
        comparison = "Per unit increase"
    else:
        comparison = "Yes vs No"

    return variable_clean, comparison


def clean_subgroup_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(MISSING_VALUES, pd.NA)

    numeric_cols = ["OR", "CI_lower", "CI_upper", "p_value", "n", "events", "non_events", "epv"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[["Variable", "Comparison"]] = df["term"].apply(
        lambda x: pd.Series(clean_term(x))
    )

    df["Subgroup variable"] = df["strat_col"].replace(TERM_MAPPING)
    df["Subgroup"] = df["strat_value"].replace(LEVEL_MAPPING)

    df["Odds ratio (95% CI)"] = (
        df["OR"].round(2).map(lambda x: f"{x:.2f}")
        + " ("
        + df["CI_lower"].round(2).map(lambda x: f"{x:.2f}")
        + "–"
        + df["CI_upper"].round(2).map(lambda x: f"{x:.2f}")
        + ")"
    )

    df["P-value"] = df["p_value"].apply(format_p_value)

    final_df = df[
        [
            "Subgroup variable",
            "Subgroup",
            "Variable",
            "Comparison",
            "Odds ratio (95% CI)",
            "P-value",
            "n",
            "events",
            "epv",
            "status",
            "suspicious_estimate",
        ]
    ].copy()

    final_df = final_df.drop_duplicates().reset_index(drop=True)

    full_subgroup_variable = final_df["Subgroup variable"].copy()
    full_subgroup = final_df["Subgroup"].copy()
    full_variable = final_df["Variable"].copy()

    final_df["Subgroup variable"] = final_df["Subgroup variable"].where(
        final_df["Subgroup variable"] != final_df["Subgroup variable"].shift(),
        "",
    )

    final_df["Subgroup"] = final_df["Subgroup"].where(
        final_df["Subgroup"] != final_df["Subgroup"].shift(),
        "",
    )

    repeated_variable = (
        final_df["Variable"].eq(final_df["Variable"].shift())
        & final_df["Subgroup"].eq("")
    )

    final_df.loc[repeated_variable, "Variable"] = ""

    # Never allow first row display labels to be blank
    final_df.loc[0, "Subgroup variable"] = full_subgroup_variable.iloc[0]
    final_df.loc[0, "Subgroup"] = full_subgroup.iloc[0]
    final_df.loc[0, "Variable"] = full_variable.iloc[0]

    final_df = final_df.drop(columns=["status", "suspicious_estimate"])

    return final_df


def main() -> None:
    raw_df = pd.read_csv(INPUT_CSV)
    cleaned_df = clean_subgroup_results(raw_df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved cleaned subgroup results to: {OUTPUT_CSV}")
    print(cleaned_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()