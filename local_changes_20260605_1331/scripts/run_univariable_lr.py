"""Run univariable logistic regression for the ESBL analysis.

This script is intentionally thin: it parses command-line options, loads the
analysis dataframe/config, then delegates the reusable regression work to
``src/utils/lr_helper_functions.py``. The main output, ``lr_data.csv``, is the
clean LR-ready dataframe used by the multivariable model script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import lr_helper_functions as lrh  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "analysis_df.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "univariable_lr"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the univariable LR workflow."""
    parser = argparse.ArgumentParser(
        description="Run univariable logistic regression from the analysis dataframe."
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
        help="Directory for output CSV files. Defaults to outputs/univariable_lr/.",
    )
    parser.add_argument(
        "--include-non-surgical",
        action="store_true",
        help="Do not restrict to surgery_before_infection == 1.",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.2,
        help="P-value threshold for candidate variable selection.",
    )
    parser.add_argument(
        "--skip-notebook-recodes",
        action="store_true",
        help="Skip prophylaxis, ethnicity, and gender recodes used in the notebook.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root unless already absolute."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict:
    """Load the project YAML config using the shared repo config loader."""
    try:
        from utils import data_cleaning_tools as dct
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the repo config loader. Make sure the project "
            "environment is active and dependencies such as PyYAML are installed."
        ) from exc

    return dct.load_config(config_path)


def write_outputs(
    output_dir: Path,
    uni_lr_df: pd.DataFrame,
    continuous_results: pd.DataFrame,
    binary_results: pd.DataFrame,
    categorical_results: pd.DataFrame,
    combined_results: pd.DataFrame,
    selected_variables: list[str],
) -> None:
    """Write LR-ready data, per-type results, combined results, and screened variables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    uni_lr_df.to_csv(output_dir / "lr_data.csv", index=False)
    continuous_results.to_csv(output_dir / "univariable_lr_continuous_results.csv", index=False)
    binary_results.to_csv(output_dir / "univariable_lr_binary_results.csv", index=False)
    categorical_results.to_csv(output_dir / "univariable_lr_categorical_results.csv", index=False)
    combined_results.to_csv(output_dir / "univariable_lr_results.csv", index=False)

    selected_df = pd.DataFrame({"predictor": selected_variables})
    selected_df.to_csv(output_dir / "variables_for_multivariable_lr.csv", index=False)


def main() -> None:
    """Run the univariable LR workflow from raw analysis dataframe to CSV outputs."""
    args = parse_args()
    config_path = resolve_path(args.config)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Loading config from %s", config_path)
    cfg = load_config(config_path)
    lr_config = cfg["logistic_regression"]

    logging.info("Loading analysis data from %s", input_path)
    analysis_df = lrh.load_analysis_data(
        input_path,
        surgery_only=not args.include_non_surgical,
    )

    if not args.skip_notebook_recodes:
        analysis_df = lrh.apply_prophylaxis_ethnicity_gender_maps(analysis_df)

    lrh.validate_lr_columns(analysis_df, lr_config)
    uni_lr_df = lrh.prepare_lr_df(analysis_df, lr_config)

    outcome = lr_config["outcome"]
    predictor_groups = lrh.get_predictor_groups(lr_config)
    reference_levels = lr_config.get("reference_levels", {})

    continuous_results = lrh.run_univariable_numeric_lr(
        uni_lr_df=uni_lr_df,
        variables=predictor_groups["continuous"],
        outcome=outcome,
        model_type="continuous",
    )
    binary_results = lrh.run_univariable_numeric_lr(
        uni_lr_df=uni_lr_df,
        variables=predictor_groups["binary"],
        outcome=outcome,
        model_type="binary",
    )
    categorical_results = lrh.run_univariable_categorical_lr(
        uni_lr_df=uni_lr_df,
        categorical_vars=predictor_groups["categorical"],
        outcome=outcome,
        reference_levels=reference_levels,
    )
    combined_results = pd.concat(
        [continuous_results, binary_results, categorical_results],
        ignore_index=True,
    ).sort_values(["model_type", "predictor", "p_value"], na_position="last")

    selected_variables = lrh.select_variables_for_multivariable_lr(
        combined_results,
        p_threshold=args.p_threshold,
    )

    write_outputs(
        output_dir=output_dir,
        uni_lr_df=uni_lr_df,
        continuous_results=continuous_results,
        binary_results=binary_results,
        categorical_results=categorical_results,
        combined_results=combined_results,
        selected_variables=selected_variables,
    )

    failed = combined_results["error"].notna().sum()
    logging.info("Wrote univariable LR outputs to %s", output_dir)
    logging.info("Rows fitted: %s; failed model rows: %s", len(combined_results), failed)
    logging.info("Selected variables at p < %.3f: %s", args.p_threshold, selected_variables)


if __name__ == "__main__":
    main()
