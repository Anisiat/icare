from pathlib import Path
import statsmodels.api as sm


def prepare_subgroup_analysis_df(df):

    df = df.copy()

    df = df.drop(
        columns=[
            "prophylaxis_group",
            "asthma",
            "copd",
            "hypertension",
            "type2_diabetes",
            "age_at_admission",
            "days_from_admission_to_first_culture",
            "crp",
            "temp",
            "colonisation"
        ],
        errors="ignore"
    )

    df = df[
        ~df["organism_bug"].isin([
            "klebsiella oxytoca",
            "proteus mirabilis"
        ])
    ]

    return df


def fit_stratified_lr(
    df,
    strat_col,
    strat_value,
    outcome_col="esbl_status",
    categorical_cols=None,
    reference_categories=None,
    min_events=10
):

    import numpy as np
    import pandas as pd

    df = df.copy()

    if categorical_cols is None:
        categorical_cols = [
            "ethnicity_desc",
            "organism_bug",
            "prophylaxis_group",
            "site",
            "tfc"
        ]

    if reference_categories is None:
        reference_categories = {
            "organism_bug": "escherichia coli",
            "site": "blood",
            "ethnicity_desc": "white",
            "prophylaxis_group": "cefuroxime + metronidazole",
            "tfc": "general_surgery"
        }

    df_stratum = df[df[strat_col] == strat_value].copy()

    y = df_stratum[outcome_col]

    if y.dtype == "object":
        y = y.map({"ESBL": 1, "non-ESBL": 0})

    X = df_stratum.drop(
        columns=[outcome_col, strat_col],
        errors="ignore"
    )

    cat_cols = [
        c for c in categorical_cols
        if c in X.columns and c != strat_col
    ]

    X = pd.get_dummies(
        X,
        columns=cat_cols,
        drop_first=False,
        prefix_sep="__",
        dtype=int
    )

    for col, ref in reference_categories.items():
        if col != strat_col:
            X = X.drop(
                columns=[f"{col}__{ref}"],
                errors="ignore"
            )

    X = X.apply(pd.to_numeric, errors="coerce")

    model_df = X.copy()
    model_df[outcome_col] = y.values
    model_df = model_df.dropna()

    y = model_df[outcome_col]
    X = model_df.drop(columns=[outcome_col])

    X = X.loc[:, X.nunique() > 1]

    n_events = int(y.sum())
    n_nonevents = int((y == 0).sum())

    if n_events < min_events or n_nonevents < min_events:
        raise ValueError(
            f"Too few events/non-events in {strat_col}={strat_value}. "
            f"Events={n_events}, non-events={n_nonevents}."
        )

    X = sm.add_constant(X, has_constant="add")

    print("Shape:", X.shape)
    print("Events:", y.sum())
    print("non-events:", (y == 0).sum())

    constant_cols = [
        c for c in X.columns
        if X[c].nunique() <= 1
    ]
    print("constant cols", constant_cols)

    dummy_cols = [c for c in X.columns if "__" in c]

    rare_cols = [
        c for c in dummy_cols
        if X[c].sum() < 5
    ]
    print("rare dummy cols:", rare_cols)

    # remove rare columns

    constant_cols = [
        c for c in X.columns
        if c != "const" and X[c].nunique() <= 1
    ]

    rare_cols = [
        c for c in X.columns
        if "__" in c and X[c].sum() < 5
    ]

    cols_to_drop = list(set(constant_cols + rare_cols))

    X = X.drop(
        columns=cols_to_drop,
        errors="ignore"
    )


    model = sm.Logit(y, X).fit(disp=False)

    converged = model.mle_retvals.get("converged", None)

    conf = model.conf_int()

    or_df = pd.DataFrame({
        "term": model.params.index,
        "OR": np.exp(model.params),
        "CI_lower": np.exp(conf[0]),
        "CI_upper": np.exp(conf[1]),
        "p_value": model.pvalues
    }).reset_index(drop=True)

    return {
    "strat_col": strat_col,
    "strat_value": strat_value,
    "n": len(model_df),
    "events": n_events,
    "non_events": n_nonevents,
    "converged": converged,
    "dropped_cols": "; ".join(cols_to_drop),
    "n_dropped_cols": len(cols_to_drop),
    "model": model,
    "or_df": or_df
    }


def run_lr_for_all_strata(
    df,
    strat_col,
    min_events=10,
    min_n=30,
    **kwargs
):
    import pandas as pd
    import warnings
    

    results = []
    errors = []

    strat_values = sorted(df[strat_col].dropna().unique())

    for strat_value in strat_values:
        print(f"\nRunning {strat_col.capitalize()}: {strat_value}")

        df_strat = df[df[strat_col] == strat_value]

        n = len(df_strat)
        

        if df_strat["esbl_status"].dtype == "object":
            n_events = (df_strat["esbl_status"] == "ESBL").sum()
            n_nonevents = (df_strat["esbl_status"] == "non-ESBL").sum()
        else:
            n_events = int(df_strat["esbl_status"].sum())
            n_nonevents = int((df_strat["esbl_status"] == 0).sum())

        if n < min_n:
            errors.append({
                'strat_col': strat_col,
                "strat_value": strat_value,
                "n": n,
                "events": n_events,
                "non_events": n_nonevents,
                "status": "skipped",
                "error_type": "too_few_rows",
                "error_message": f"Only {n} rows"
            })
            continue

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                result = fit_stratified_lr(
                df=df,
                strat_col=strat_col,
                strat_value=strat_value,
                min_events=min_events,
                **kwargs
            )

                warning_messages = [
                    str(w.message) for w in caught_warnings
                ]

            or_df = result["or_df"].copy()
            or_df["strat_col"] = strat_col
            or_df["strat_value"] = strat_value
            or_df["n"] = result["n"]
            or_df["events"] = result["events"]
            or_df["non_events"] = result["non_events"]
            or_df["status"] = "success"
            or_df["warnings"] = "; ".join(warning_messages)
            or_df["converged"] = result["converged"]
            or_df["dropped_cols"] = result["dropped_cols"]
            or_df["n_dropped_cols"] = result["n_dropped_cols"]

            or_df["suspicious_estimate"] = (
            (or_df["OR"] > 100) |
            (or_df["CI_upper"] > 1000) |
            (
            (or_df["CI_lower"] > 0) &
            ((or_df["CI_upper"] / or_df["CI_lower"]) > 100)
            )
                )

            results.append(or_df)

            if warning_messages:
                errors.append({
                    "strat_col": strat_col,
                    "strat_value": strat_value,
                    "n": result["n"],
                    "events": result["events"],
                    "non_events": result["non_events"],
                    "status": "warning",
                    "error_type": "model_warning",
                    "error_message": "; ".join(warning_messages)
                })

        except Exception as e:
            errors.append({
                "strat_col": strat_col,
                "strat_value": strat_value,
                "n": n,
                "events": n_events,
                "non_events": n_nonevents,
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

    if results:
        results_df = pd.concat(results, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    errors_df = pd.DataFrame(errors)

    return results_df, errors_df


def main(cfg_path="config.yaml"):

    from src.utils import load_config
    import pandas as pd

    config = load_config(cfg_path)
    df = pd.read_csv(config["paths"]["lr_dataset"])

    sub_analysis_df = prepare_subgroup_analysis_df(df)

    results_tfc, errors_tfc = run_lr_for_all_strata(sub_analysis_df, strat_col="tfc")
    results_site, errors_site = run_lr_for_all_strata(sub_analysis_df, strat_col="site")

    results_df = pd.concat([results_tfc, results_site], ignore_index=True)
    errors_df = pd.concat([errors_tfc, errors_site], ignore_index=True)

    Path("outputs/subgroup_analysis").mkdir(
    parents=True,
    exist_ok=True
    )
    
    results_df.to_csv("outputs/subgroup_analysis/subgroup_analysis_lr_results.csv", index=False)
    errors_df.to_csv("outputs/subgroup_analysis/subgroup_analysis_lr_errors.csv", index=False)
    

if __name__ == "__main__":
    
    main()


    