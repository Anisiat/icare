"""Create Table 1 for the ESBL analysis.

This script is a command-line version of the Table 1 workflow in
notebooks/LR_analysis_refactored.ipynb. By default it reads the analysis
dataset from data/interim/analysis_df.csv and writes outputs to outputs/table1/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "analysis_df.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "table1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Table 1 from the analysis dataframe."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML. Defaults to configs/config.yaml.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to analysis_df CSV. Defaults to data/interim/analysis_df.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files. Defaults to outputs/table1/.",
    )
    parser.add_argument(
        "--include-non-surgical",
        action="store_true",
        help="Do not restrict to surgery_before_infection == 1.",
    )
   
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Do not write the LaTeX table.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the TableOne object to stdout.",
    )
    parser.add_argument(
        "--first_episode_only",
        action="store_true",
        help="Only include the first episode for each subject in Table 1 (instead of all episodes).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict:
    try:
        from utils import data_cleaning_tools as dct
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the repo config loader. Make sure the project "
            "environment is active and dependencies such as PyYAML are installed."
        ) from exc

    return dct.load_config(config_path)


def clean_missing(value: object) -> object:
    """Convert common string encodings of missing/unknown values to np.nan."""
    if isinstance(value, str):
        value_clean = value.strip().lower()
        if value_clean in {
            "unknown",
            "other",
            "other_unknown",
            "none",
            "nan",
            "",
        }:
            return np.nan

    return value


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "map"):
        return df.map(clean_missing)

    return df.applymap(clean_missing)


def load_analysis_data(path: Path, surgery_only: bool = True) -> pd.DataFrame:
    analysis_df = pd.read_csv(path)

    if surgery_only:
        if "surgery_before_infection" not in analysis_df.columns:
            raise KeyError(
                "Cannot filter to surgical patients because "
                "'surgery_before_infection' is missing."
            )
        analysis_df = analysis_df[analysis_df["surgery_before_infection"] == 1].copy()

    return analysis_df


def regroup_prophylaxis(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    value = str(value).lower().strip()

    if "cefuroxime" in value and "metronidazole" in value:
        return "cefuroxime + metronidazole"
    if "co-amoxiclav" in value:
        return "co-amoxiclav-based"
    if "glycopeptide" in value or "teicoplanin" in value or "vancomycin" in value:
        return "glycopeptide-based"
    if "aminoglycoside" in value or "gentamicin" in value:
        return "aminoglycoside-based"
    if "clindamycin" in value:
        return "clindamycin-based"
    if "metronidazole" in value:
        return "metronidazole-only/+other"
    if "cefuroxime" in value:
        return "cefuroxime-only/+other"
    if "piperacillin-tazobactam" in value:
        return "broad-spectrum beta-lactam"

    return "other"


def apply_prophylaxis_ethnicity_gender_maps(analysis_df: pd.DataFrame) -> pd.DataFrame:
    analysis_df = analysis_df.copy()

    if "prophylaxis_group" in analysis_df.columns:
        analysis_df["prophylaxis_group"] = analysis_df["prophylaxis_group"].apply(
            regroup_prophylaxis
        )

    if "ethnicity_desc" in analysis_df.columns:
        analysis_df["ethnicity_desc"] = analysis_df["ethnicity_desc"].replace(
            {
                "mixed - any other mixed background": "mixed",
                "mixed - white and black caribbean": "mixed",
                "mixed - white and asian": "mixed",
                "mixed - white and black african": "mixed",
                "mixed": "mixed",
            }
        )

    if "gender" in analysis_df.columns:
        non_missing_gender = set(analysis_df["gender"].dropna().unique())
        if non_missing_gender and non_missing_gender.issubset({1, 2}):
            # maps male to 0 and female to 1
            analysis_df["gender"] = analysis_df["gender"].map({2: 1, 1: 0})

    return analysis_df.drop(columns="prophylaxis_class", errors="ignore")


def validate_table1_columns(analysis_df: pd.DataFrame, table1_config: dict) -> None:
    required_cols = table1_config["columns"]
    missing_cols = [col for col in required_cols if col not in analysis_df.columns]

    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Analysis dataframe is missing Table 1 columns: {missing}")


def get_numeric_columns(table1_config: dict) -> list[str]:
    numeric_cols = list(table1_config.get("nonnormal", []))
    numeric_cols.extend(
        [
            "imd_decile",
            "age_at_admission",
            "length_of_stay_days",
            "temp",
            "crp",
        ]
    )

    return list(dict.fromkeys(numeric_cols))


def get_table1_df(analysis_df: pd.DataFrame, table1_config: dict) -> pd.DataFrame:
    table1_df = analysis_df[table1_config["columns"]].copy()
    table1_df = clean_missing_values(table1_df)

    for col in get_numeric_columns(table1_config):
        if col in table1_df.columns:
            table1_df[col] = pd.to_numeric(table1_df[col], errors="coerce")

    return table1_df


def groupby_first_episode(table1_df: pd.DataFrame) -> pd.DataFrame:

    # Get table one either for all episodes, or stratified/group by subject or first episode 
    table1_df = table1_df.copy()

    table1_df = table1_df.sort_values("admission_date").groupby("subject").first().reset_index()

    return table1_df



def create_table1(table1_df: pd.DataFrame, table1_config: dict):
    try:
        from tableone import TableOne
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import tableone. Make sure the project environment is "
            "active and the tableone package is installed."
        ) from exc

    columns = [col for col in table1_config["columns"] if col in table1_df.columns]
    categorical = [
        col for col in table1_config.get("categorical", []) if col in table1_df.columns
    ]
    nonnormal = [
        col for col in table1_config.get("nonnormal", []) if col in table1_df.columns
    ]

    return TableOne(
        data=table1_df,
        columns=columns,
        categorical=categorical,
        groupby=table1_config["groupby"],
        nonnormal=nonnormal,
        pval=True,
        missing=True,
        include_null=False,
    )


def tableone_to_dataframe(table1) -> pd.DataFrame:
    if hasattr(table1, "tableone"):
        return table1.tableone.copy()

    return pd.DataFrame(str(table1).splitlines())


def write_outputs(
    output_dir: Path,
    table1_df: pd.DataFrame,
    table1,
    write_latex: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table1_df.to_csv(output_dir / "table1_data.csv", index=False)
    tableone_to_dataframe(table1).to_csv(output_dir / "table1_epidemiology_esbl.csv")

    if write_latex:
        table1.to_latex(output_dir / "table1_epidemiology_esbl.tex")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Loading config from %s", config_path)
    cfg = load_config(config_path)
    table1_config = cfg["table1"]

    logging.info("Loading analysis data from %s", input_path)
    analysis_df = load_analysis_data(
        input_path,
        surgery_only=not args.include_non_surgical,
    )

    validate_table1_columns(analysis_df, table1_config)

    analysis_df = apply_prophylaxis_ethnicity_gender_maps(analysis_df)

    if args.first_episode_only:
        required = {"subject", "admission_date"}
        missing = required - set(analysis_df.columns)
        if missing:
            raise KeyError(f"Missing columns needed for first episode selection: {missing}")
        analysis_df = groupby_first_episode(analysis_df)

    table1_df = get_table1_df(analysis_df, table1_config)
    table1 = create_table1(table1_df, table1_config)

    write_outputs(
        output_dir=output_dir,
        table1_df=table1_df,
        table1=table1,
        write_latex=not args.no_latex,
    )

    if args.print:
        print(table1)

    logging.info("Wrote Table 1 outputs to %s", output_dir)


if __name__ == "__main__":
    main()
