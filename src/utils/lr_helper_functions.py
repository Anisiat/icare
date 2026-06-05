"""Helper functions for logistic regression analysis scripts.

The functions here are shared by the univariable and multivariable LR command
line scripts. They keep config parsing, notebook-style recoding, model dataframe
construction, model fitting, and diagnostics in one place so the scripts can
stay small and readable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def remove_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common unknown/missing string labels with ``pd.NA``."""
    unknown_values = [
        "other",
        "other_unknown",
        "unknown",
        "unk",
        "not known",
        "not specified",
        "missing",
        "none",
        "",
    ]

    return df.copy().replace(to_replace=unknown_values, value=pd.NA)


def load_analysis_data(path: Path, surgery_only: bool = True) -> pd.DataFrame:
    """Load ``analysis_df`` and optionally keep only surgical infection episodes."""
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
    """Collapse raw prophylaxis strings into broader analysis groups."""
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
    """Apply the recodes used in the LR notebook before modelling.

    This currently regroups prophylaxis, collapses mixed ethnicity labels to
    missing for LR, and maps the raw gender coding from 1/2 to 0/1 when present.
    """
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
                "mixed": pd.NA,
            }
        )

    if "gender" in analysis_df.columns:
        non_missing_gender = set(analysis_df["gender"].dropna().unique())
        if non_missing_gender and non_missing_gender.issubset({1, 2}):
            analysis_df["gender"] = analysis_df["gender"].map({2: 1, 1: 0})

    return analysis_df


def get_predictor_groups(lr_config: dict) -> dict[str, list[str]]:
    """Return continuous, binary, and categorical predictor lists from config."""
    predictors = lr_config.get("predictors")

    if not isinstance(predictors, dict):
        raise KeyError(
            "Expected logistic_regression.predictors in the config, with "
            "continuous, binary, and categorical lists."
        )

    return {
        "continuous": predictors.get("continuous", []),
        "binary": predictors.get("binary", []),
        "categorical": predictors.get("categorical", []),
    }


def get_configured_predictors(lr_config: dict) -> list[str]:
    """Return all configured model predictors, preserving config order."""
    predictor_groups = get_predictor_groups(lr_config)
    predictors = (
        predictor_groups["continuous"]
        + predictor_groups["binary"]
        + predictor_groups["categorical"]
    )

    return list(dict.fromkeys(predictors))


def get_lr_columns(lr_config: dict) -> tuple[list[str], list[str]]:
    """Return columns to keep in LR data and columns to use as predictors.

    The kept columns include ID columns and the outcome so later scripts can
    cluster, select first episodes, or join back to source data. Predictor
    columns exclude IDs and outcome.
    """
    predictor_cols = get_configured_predictors(lr_config)
    id_cols = lr_config.get("id_columns", [])
    outcome = lr_config["outcome"]

    keep_cols = id_cols + [outcome] + predictor_cols
    keep_cols = list(dict.fromkeys(keep_cols))

    return keep_cols, predictor_cols


def clean_predictor_list(predictors: list[str], lr_config: dict) -> list[str]:
    """Remove ID and outcome columns from a proposed predictor list."""
    excluded = set(lr_config.get("id_columns", []))
    excluded.add(lr_config["outcome"])

    return list(dict.fromkeys([col for col in predictors if col not in excluded]))


def load_predictors_from_file(path: Path) -> list[str]:
    """Load a one-column predictor list from a CSV file."""
    variables_df = pd.read_csv(path)

    if "predictor" not in variables_df.columns:
        raise KeyError(f"{path} must contain a 'predictor' column.")

    return variables_df["predictor"].dropna().astype(str).tolist()


def validate_lr_columns(analysis_df: pd.DataFrame, lr_config: dict) -> None:
    """Raise a clear error if configured LR columns are absent from the data."""
    keep_cols, _ = get_lr_columns(lr_config)
    missing_cols = [col for col in keep_cols if col not in analysis_df.columns]

    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Analysis dataframe is missing LR columns from config: {missing}")


def encode_outcome(series: pd.Series) -> pd.Series:
    """Encode ESBL outcome values as 1/0, accepting text or numeric inputs."""
    if series.dtype == "object" or str(series.dtype).startswith("string"):
        return series.map({"ESBL": 1, "non-ESBL": 0})

    return pd.to_numeric(series, errors="coerce")


def prepare_lr_df(analysis_df: pd.DataFrame, lr_config: dict) -> pd.DataFrame:
    """Create the LR-ready dataframe used by univariable and adjusted models.

    ID columns are retained for clustering/sensitivity analyses, but only
    configured predictors are coerced for modelling. Categorical predictors are
    left as pandas categories for formula-based univariable models.
    """
    keep_cols, _ = get_lr_columns(lr_config)
    predictor_groups = get_predictor_groups(lr_config)
    lr_df = analysis_df[keep_cols].copy()

    outcome = lr_config["outcome"]
    continuous_vars = predictor_groups["continuous"]
    binary_vars = predictor_groups["binary"]
    categorical_vars = predictor_groups["categorical"]

    lr_df = remove_unknowns(lr_df)
    lr_df[outcome] = encode_outcome(lr_df[outcome])

    for col in continuous_vars + binary_vars:
        if col in lr_df.columns:
            lr_df[col] = pd.to_numeric(lr_df[col], errors="coerce")

    for col in categorical_vars:
        if col in lr_df.columns:
            lr_df[col] = lr_df[col].astype("category")

    return lr_df


def run_univariable_numeric_lr(
    uni_lr_df: pd.DataFrame,
    variables: list[str],
    outcome: str,
    model_type: str,
) -> pd.DataFrame:
    """Fit one univariable logistic model per numeric or binary predictor."""
    import statsmodels.formula.api as smf

    results = []

    for var in variables:
        model_df = uni_lr_df[[outcome, var]].dropna()

        try:
            model = smf.logit(f"{outcome} ~ {var}", data=model_df).fit(disp=0)
            conf = model.conf_int().loc[var]
            coef = model.params[var]

            results.append(
                {
                    "predictor": var,
                    "term": var,
                    "category": pd.NA,
                    "reference": pd.NA,
                    "odds_ratio": np.exp(coef),
                    "ci_lower": np.exp(conf[0]),
                    "ci_upper": np.exp(conf[1]),
                    "p_value": model.pvalues[var],
                    "n": len(model_df),
                    "events": int(model_df[outcome].sum()),
                    "model_type": model_type,
                    "error": pd.NA,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "predictor": var,
                    "term": var,
                    "category": pd.NA,
                    "reference": pd.NA,
                    "odds_ratio": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "n": len(model_df),
                    "events": int(model_df[outcome].sum()) if len(model_df) else 0,
                    "model_type": model_type,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(results)


def run_univariable_categorical_lr(
    uni_lr_df: pd.DataFrame,
    categorical_vars: list[str],
    outcome: str,
    reference_levels: dict[str, str],
) -> pd.DataFrame:
    """Fit one univariable logistic model per categorical predictor.

    Each non-reference category is returned as a separate odds-ratio row. If a
    configured reference level is absent in complete cases, statsmodels chooses
    its default reference and the output reference is recorded as missing.
    """
    import statsmodels.formula.api as smf

    results = []

    for var in categorical_vars:
        model_df = uni_lr_df[[outcome, var]].dropna()
        reference = reference_levels.get(var)

        if reference is not None and reference in set(model_df[var].dropna()):
            formula = f"{outcome} ~ C({var}, Treatment(reference={reference!r}))"
        else:
            formula = f"{outcome} ~ C({var})"
            reference = pd.NA

        try:
            model = smf.logit(formula, data=model_df).fit(disp=0)
            conf = model.conf_int()

            for term in model.params.index:
                if term == "Intercept":
                    continue

                category = term.split("[T.")[-1].rstrip("]")
                coef = model.params[term]

                results.append(
                    {
                        "predictor": var,
                        "term": term,
                        "category": category,
                        "reference": reference,
                        "odds_ratio": np.exp(coef),
                        "ci_lower": np.exp(conf.loc[term, 0]),
                        "ci_upper": np.exp(conf.loc[term, 1]),
                        "p_value": model.pvalues[term],
                        "n": len(model_df),
                        "events": int(model_df[outcome].sum()),
                        "model_type": "categorical",
                        "error": pd.NA,
                    }
                )
        except Exception as exc:
            results.append(
                {
                    "predictor": var,
                    "term": pd.NA,
                    "category": pd.NA,
                    "reference": reference,
                    "odds_ratio": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "n": len(model_df),
                    "events": int(model_df[outcome].sum()) if len(model_df) else 0,
                    "model_type": "categorical",
                    "error": str(exc),
                }
            )

    return pd.DataFrame(results)


def select_variables_for_multivariable_lr(
    results_df: pd.DataFrame,
    p_threshold: float,
    clinically_important_predictors: list[str] | None = None,
) -> list[str]:
    """Select predictors for adjusted modelling using p-value screening plus defaults."""
    if clinically_important_predictors is None:
        clinically_important_predictors = ["age_at_admission", "gender"]

    screened = results_df.loc[
        results_df["p_value"].notna() & (results_df["p_value"] < p_threshold),
        "predictor",
    ].tolist()

    return sorted(set(screened + clinically_important_predictors))


def restrict_to_first_episode(
    df: pd.DataFrame,
    subject_col: str = "subject",
    episode_col: str = "infection_id",
) -> pd.DataFrame:
    """Keep the earliest infection episode per subject for sensitivity analyses."""
    if subject_col not in df.columns:
        raise KeyError(f"Cannot select first episodes because '{subject_col}' is missing.")

    sort_cols = [subject_col]
    if episode_col in df.columns:
        sort_cols.append(episode_col)

    return df.sort_values(sort_cols).drop_duplicates(subject_col, keep="first").copy()


def validate_model_columns(df: pd.DataFrame, outcome: str, predictors: list[str]) -> None:
    """Raise a clear error if adjusted model columns are absent from LR data."""
    missing_cols = [col for col in [outcome] + predictors if col not in df.columns]

    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Input dataframe is missing multivariable LR columns: {missing}")


def get_model_dataframe(
    df: pd.DataFrame,
    lr_config: dict,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build complete-case ``model_df``, encoded predictor matrix ``X``, and outcome ``y``.

    Categorical predictors are dummy-encoded with ``<variable>__<category>``
    column names. Configured reference levels are dropped after encoding so the
    remaining coefficients compare against those references.
    """
    outcome = lr_config["outcome"]
    categorical_cols = [
        col for col in get_predictor_groups(lr_config)["categorical"] if col in predictors
    ]
    reference_levels = lr_config.get("reference_levels", {})

    validate_model_columns(df, outcome, predictors)

    y = encode_outcome(df[outcome])
    X = df[predictors].copy()

    X = pd.get_dummies(
        X,
        columns=[col for col in categorical_cols if col in X.columns],
        prefix_sep="__",
        drop_first=False,
        dtype=int,
    )

    for col, ref in reference_levels.items():
        X = X.drop(columns=f"{col}__{ref}", errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")

    model_df = X.copy()
    model_df[outcome] = y.values
    model_df = model_df.dropna()

    y_model = model_df[outcome].astype(int)
    X_model = model_df.drop(columns=[outcome])
    X_model = X_model.loc[:, X_model.nunique(dropna=False) > 1]

    return model_df[[outcome] + list(X_model.columns)], X_model, y_model


def calculate_epv(y: pd.Series, n_predictors: int) -> pd.DataFrame:
    """Calculate events-per-variable using the smaller outcome class count."""
    events = int(y.sum())
    non_events = int((y == 0).sum())
    limiting_events = min(events, non_events)
    epv = limiting_events / n_predictors if n_predictors else np.nan

    return pd.DataFrame(
        [
            {
                "n_complete_cases": len(y),
                "events": events,
                "non_events": non_events,
                "n_predictors": n_predictors,
                "epv": epv,
            }
        ]
    )


def get_correlation_screen(
    X: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return absolute predictor correlations and pairs above a threshold."""
    corr_matrix = X.corr().abs().fillna(0)
    pairs = []
    columns = list(corr_matrix.columns)

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            corr = corr_matrix.loc[col_a, col_b]
            if corr >= threshold:
                pairs.append(
                    {
                        "variable_1": col_a,
                        "variable_2": col_b,
                        "abs_correlation": corr,
                    }
                )

    high_corr_df = pd.DataFrame(
        pairs,
        columns=["variable_1", "variable_2", "abs_correlation"],
    )

    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values(
            "abs_correlation",
            ascending=False,
            ignore_index=True,
        )

    return corr_matrix, high_corr_df


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    save_path: str | Path | None = "correlation_heatmap.png",
) -> None:
    """Plot and optionally save a heatmap of the predictor correlation matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.5})
    plt.title("Predictor Correlation Matrix", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance inflation factors for encoded model predictors."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_numeric = X.apply(pd.to_numeric, errors="coerce").dropna()
    X_numeric = X_numeric.loc[:, X_numeric.nunique(dropna=False) > 1]

    if X_numeric.empty:
        return pd.DataFrame(columns=["variable", "vif", "error"])

    X_with_const = sm.add_constant(X_numeric, has_constant="add")

    rows = []
    for i, variable in enumerate(X_with_const.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
            rows.append({"variable": variable, "vif": vif, "error": pd.NA})
        except Exception as exc:
            rows.append({"variable": variable, "vif": np.nan, "error": str(exc)})

    return pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last")


def split_encoded_term(term: str) -> tuple[str, str]:
    """Split a dummy-encoded term into original predictor and category labels."""
    if "__" in term:
        predictor, category = term.split("__", 1)
        return predictor, category

    return term, ""


def fit_multivariable_lr(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
) -> tuple[object, pd.DataFrame]:
    """Fit an adjusted logistic regression model and return a tidy OR table.

    If ``groups`` is provided, statsmodels uses cluster-robust standard errors
    with those group labels, which is useful for repeated infection episodes
    clustered within patients.
    """
    import statsmodels.api as sm

    X_with_const = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X_with_const)

    if groups is None:
        results = model.fit(disp=0)
    else:
        results = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})

    conf = results.conf_int()

    results_df = pd.DataFrame(
        {
            "term": results.params.index,
            "odds_ratio": np.exp(results.params),
            "ci_lower": np.exp(conf[0]),
            "ci_upper": np.exp(conf[1]),
            "p_value": results.pvalues,
        }
    ).reset_index(drop=True)

    results_df = results_df[results_df["term"] != "const"].copy()
    results_df[["predictor", "category"]] = results_df["term"].apply(
        lambda term: pd.Series(split_encoded_term(term))
    )
    results_df["is_categorical"] = results_df["category"] != ""
    results_df["n"] = len(y)
    results_df["events"] = int(y.sum())

    return results, results_df


def get_model_summary(model) -> pd.DataFrame:
    """Return convergence and fit statistics for a fitted statsmodels model."""
    mle_retvals = getattr(model, "mle_retvals", {})

    return pd.DataFrame(
        [
            {
                "converged": mle_retvals.get("converged", pd.NA),
                "iterations": mle_retvals.get("iterations", pd.NA),
                "llf": getattr(model, "llf", np.nan),
                "aic": getattr(model, "aic", np.nan),
                "bic": getattr(model, "bic", np.nan),
                "pseudo_r2": getattr(model, "prsquared", np.nan),
            }
        ]
    )


def plot_adjusted_or_forest(
    or_df: pd.DataFrame,
    max_ci_upper: float = 20,
    save_path: str | Path | None = "figure_poster.png",
    include_groups: list[str] | None = None,
    exclude_groups: list[str] | None = None,
    include_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    show_only_significant: bool = False,
) -> pd.DataFrame:
    """Create a forest plot from adjusted LR odds-ratio results.

    The input may use either ``odds_ratio`` or ``OR`` for the point estimate.
    The returned dataframe is the filtered/labelled data used for plotting.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plot_df = or_df.copy()

    if "OR" not in plot_df.columns and "odds_ratio" in plot_df.columns:
        plot_df["OR"] = plot_df["odds_ratio"]

    plot_df = plot_df[plot_df["predictor"] != "Intercept"].copy()
    plot_df = plot_df[
        np.isfinite(plot_df["OR"])
        & np.isfinite(plot_df["ci_lower"])
        & np.isfinite(plot_df["ci_upper"])
        & (plot_df["ci_lower"] > 0)
        & (plot_df["ci_upper"] < max_ci_upper)
    ].copy()

    category_map = {
        "asian": "Asian",
        "black": "Black",
        "white": "White",
        "escherichia_coli": "E. coli",
        "klebsiella pneumoniae": "K. pneumoniae",
        "klebsiella oxytoca": "K. oxytoca",
        "proteus mirabilis": "P. mirabilis",
        "blood": "Blood",
        "drain": "Drain",
        "wound": "Wound",
        "tips_devices": "Catheter/device",
        "tissue/biopsy": "Tissue/biopsy",
        "urine": "Urine",
        "sputum": "Sputum",
        "gynaecological_oncology": "Gynaecological oncology",
        "vascular": "Vascular surgery",
        "uro_nephro": "Urology/nephrology",
        "abdominal_gi": "Abdominal/GI surgery",
        "general_surgery": "General surgery",
        "ortho_plastics": "Orthopaedics/plastics",
        "cardiothoracic": "Cardiothoracic",
        "neuro_ent": "Neurosurgery/ENT",
        "obstetrics": "Obstetrics",
        "gynaecology": "Gynaecology",
        "aminoglycoside-based": "Aminoglycoside-based",
        "broad-spectrum beta-lactam": "Broad-spectrum beta-lactam",
        "cefuroxime-only/+other": "Cefuroxime-only / +other",
        "clindamycin-based": "Clindamycin-based",
        "co-amoxiclav-based": "Co-amoxiclav-based",
        "glycopeptide-based": "Glycopeptide-based",
        "metronidazole-only/+other": "Metronidazole-only / +other",
    }

    predictor_map = {
        "age_at_admission": "Age at admission",
        "asthma": "Asthma",
        "colonisation": "Colonisation",
        "copd": "COPD",
        "crp": "CRP",
        "days_from_admission_to_first_culture": "Days from admission to first culture",
        "gender": "Gender",
        "hospital_exposure_90d": "Hospital exposure in previous 90 days",
        "hypertension": "Hypertension",
        "past_abx": "Prior antibiotics",
        "surgery_length": "Surgery length",
        "temp": "Temperature",
        "type2_diabetes": "Type 2 diabetes",
        "organism_bug": "Organism",
        "site": "Sample site",
        "tfc": "Surgery type",
        "prophylaxis_group": "Prophylaxis",
        "ethnicity_desc": "Ethnicity",
    }

    plot_df["predictor_clean"] = plot_df["predictor"].replace(predictor_map)
    plot_df["category_clean"] = plot_df["category"].replace(category_map)
    plot_df["label"] = np.where(
        plot_df["is_categorical"],
        plot_df["category_clean"],
        plot_df["predictor_clean"],
    )
    plot_df["group"] = np.where(
        plot_df["is_categorical"],
        plot_df["predictor_clean"],
        "Clinical covariates",
    )

    if include_groups is not None:
        plot_df = plot_df[plot_df["group"].isin(include_groups)].copy()
    if exclude_groups is not None:
        plot_df = plot_df[~plot_df["group"].isin(exclude_groups)].copy()
    if include_terms is not None:
        plot_df = plot_df[plot_df["term"].isin(include_terms)].copy()
    if exclude_terms is not None:
        plot_df = plot_df[~plot_df["term"].isin(exclude_terms)].copy()
    if include_labels is not None:
        plot_df = plot_df[plot_df["label"].isin(include_labels)].copy()
    if exclude_labels is not None:
        plot_df = plot_df[~plot_df["label"].isin(exclude_labels)].copy()
    if show_only_significant:
        plot_df = plot_df[plot_df["p_value"] < 0.05].copy()

    group_order = [
        "Organism",
        "Sample site",
        "Surgery type",
        "Prophylaxis",
        "Ethnicity",
        "Clinical covariates",
    ]
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values(["group", "OR"], ascending=[True, False]).reset_index(drop=True)

    y_positions = []
    current_y = 0
    for group in group_order:
        df_group = plot_df[plot_df["group"] == group]
        for _ in range(len(df_group)):
            y_positions.append(current_y)
            current_y += 1
        if len(df_group):
            current_y += 1

    plot_df["y"] = y_positions
    plot_df["significant"] = plot_df["p_value"] < 0.05

    _, ax = plt.subplots(figsize=(10, max(8, 0.45 * len(plot_df))))
    for significant, color, label in [(False, "0.65", "p >= 0.05"), (True, "navy", "p < 0.05")]:
        df_part = plot_df[plot_df["significant"] == significant]
        ax.errorbar(
            df_part["OR"],
            df_part["y"],
            xerr=[
                df_part["OR"] - df_part["ci_lower"],
                df_part["ci_upper"] - df_part["OR"],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
            markersize=8,
            alpha=0.9,
            label=label,
        )

    for _, row in plot_df[plot_df["significant"]].iterrows():
        ax.annotate(
            f'{row["OR"]:.2f}',
            xy=(row["OR"], row["y"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            color="navy",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
        )

    ax.axvline(1, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["label"], fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Odds ratio (95% CI, log scale)", fontsize=14, labelpad=10)
    ax.set_title("Factors associated with ESBL infection", fontsize=20, weight="bold", pad=20)
    ax.grid(axis="x", linestyle=":", alpha=0.35, linewidth=1.2)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return plot_df
