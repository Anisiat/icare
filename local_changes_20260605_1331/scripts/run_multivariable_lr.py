"""Run multivariable logistic regression for the ESBL analysis."""

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
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "univariable_lr" / "lr_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multivariable_lr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multivariable logistic regression from LR-ready data."
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
        help="Path to lr_data.csv from run_univariable_lr.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV files. Defaults to outputs/multivariable_lr/.",
    )
    parser.add_argument(
        "--episodes",
        choices=["all", "first"],
        default="all",
        help="Use all infection episodes or only the first episode per subject.",
    )

    parser.add_argument(
        "--cluster-by-patient",
        action="store_true",
        help="Use patient-level cluster-robust standard errors.",
        )

    parser.add_argument(
        "--variables-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with a predictor column. If supplied, fit only those "
            "predictors instead of all configured predictors."
        ),
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.7,
        help="Absolute correlation threshold for multicollinearity screening.",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=5.0,
        help="VIF threshold for flagging possible multicollinearity.",
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


def write_outputs(
    output_dir: Path,
    model_df: pd.DataFrame,
    results_df: pd.DataFrame,
    epv_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    high_vif_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    high_corr_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_df.to_csv(output_dir / "multivariable_lr_model_data.csv", index=False)
    results_df.to_csv(output_dir / "multivariable_lr_results.csv", index=False)
    epv_df.to_csv(output_dir / "epv.csv", index=False)
    vif_df.to_csv(output_dir / "vif.csv", index=False)
    high_vif_df.to_csv(output_dir / "high_vif.csv", index=False)
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
    high_corr_df.to_csv(output_dir / "high_correlation_pairs.csv", index=False)
    summary_df.to_csv(output_dir / "model_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    variables_file = resolve_path(args.variables_file) if args.variables_file else None

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Loading config from %s", config_path)
    cfg = load_config(config_path)
    lr_config = cfg["logistic_regression"]

    logging.info("Loading LR-ready data from %s", input_path)
    lr_df = pd.read_csv(input_path)

    if args.episodes == "first":
        lr_df = lrh.restrict_to_first_episode(lr_df)
        output_dir = output_dir / "first_episodes"
    else:
        output_dir = output_dir / "all_episodes"

    if args.cluster_by_patient and args.episodes == "first":
        raise ValueError(
            "Do not use --cluster-by-patient with --episodes first. "
            "Clustering is only relevant when patients can contribute multiple episodes.")

    if args.cluster_by_patient:
        output_dir = output_dir / "clustered_by_patient"
    else:
        output_dir = output_dir / "unclustered"

    if variables_file is not None:
        predictors = lrh.load_predictors_from_file(variables_file)
    else:
        predictors = lrh.get_configured_predictors(lr_config)
    predictors = lrh.clean_predictor_list(predictors, lr_config)

    if not predictors:
        raise ValueError("No predictors available after excluding ID/outcome columns.")

    model_df, X, y = lrh.get_model_dataframe(lr_df, lr_config, predictors)

    epv_df = lrh.calculate_epv(y, n_predictors=X.shape[1])
    corr_matrix, high_corr_df = lrh.get_correlation_screen(
        X,
        threshold=args.correlation_threshold,
    )
    vif_df = lrh.calculate_vif(X)
    high_vif_df = vif_df[
        (vif_df["variable"] != "const") & (vif_df["vif"] >= args.vif_threshold)
    ].copy()

    #if clustering is required

    groups = None

    if args.cluster_by_patient:
        groups = lr_df.loc[model_df.index, "subject"]
        assert len(groups) == len(X)
        assert len(groups) == len(y)

    model, results_df = lrh.fit_multivariable_lr(X, y, groups=groups)
    
    summary_df = lrh.get_model_summary(model)

    write_outputs(
        output_dir=output_dir,
        model_df=model_df,
        results_df=results_df,
        epv_df=epv_df,
        vif_df=vif_df,
        high_vif_df=high_vif_df,
        corr_matrix=corr_matrix,
        high_corr_df=high_corr_df,
        summary_df=summary_df,
    )

    epv_row = epv_df.iloc[0]
    logging.info("Wrote multivariable LR outputs to %s", output_dir)
    logging.info(
        "Complete cases: %s; events: %s; non-events: %s; predictors: %s; EPV: %.2f",
        epv_row["n_complete_cases"],
        epv_row["events"],
        epv_row["non_events"],
        epv_row["n_predictors"],
        epv_row["epv"],
    )
    logging.info(
        "High VIF terms at >= %.2f: %s",
        args.vif_threshold,
        len(high_vif_df),
    )
    logging.info(
        "High-correlation pairs at >= %.2f: %s",
        args.correlation_threshold,
        len(high_corr_df),
    )


if __name__ == "__main__":
    main()
