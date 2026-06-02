import sys
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import re

antibiotics =  ['amikacin', 'amoxicillin', 
       'cefepime', 'cefotaxime', 'cefoxitin', 'cefpodoxime', 'ceftazidime',
       'ceftriaxone', 'cefuroxime', 'cephalexin', 'ciprofloxacin',
       'co-amoxiclav', 'cotrimoxazole', 'ertapenem', 'gentamicin', 'imipenem', 'meropenem',
       'nitrofurantoin', 'piperacillin-tazobactam', 'tazocin', 'temocillin',
       'tigecycline']




antibiogram_cols_extended = [
    'lab_test_id', 'latest_collect_dt', 'site', 'organism_bug', 'amikacin', 'amoxycillin',
    'augmentin', 'cefepime', 'cefotaxime', 'cefoxitin', 'cefpodoxime',
    'ceftazidime', 'ceftriaxone', 'cefuroxime', 'cephalexin',
    'ciprofloxacin', 'co-amoxiclav', 'cotrimoxazole', 'ertapenem', 'gentamicin', 'imipenem', 'meropenem',
    'nitrofurantoin', 'piperacillin-tazobactam', 'tazocin', 'temocillin',
    'tigecycline', 'culture_type', 'surgical_area', 'age_at_admission', 'treatment_function_code',
    'esbl_status', 'site_grouped', 'tfc_desc'
]



def get_antibiogram_df(mcs_df):
    """
    Clean microbiology data for antibiogram analysis.

    Keeps target organisms, removes conflicting duplicate results,
    and combines augmentin / co-amoxiclav into a single column.
    """

    import pandas as pd

    antibiogram_cols = [
        "lab_test_id",
        "organism_bug",
        "site_grouped",
        "amikacin",
        "amoxicillin",
        "augmentin",
        "cefepime",
        "cefotaxime",
        "cefoxitin",
        "cefpodoxime",
        "ceftazidime",
        "ceftriaxone",
        "cefuroxime",
        "cephalexin",
        "ciprofloxacin",
        "co-amoxiclav",
        "cotrimoxazole",
        "ertapenem",
        "gentamicin",
        "imipenem",
        "meropenem",
        "nitrofurantoin",
        "piperacillin-tazobactam",
        "tazocin",
        "temocillin",
        "tigecycline",
    ]

    target_organisms = [
        "escherichia coli",
        "klebsiella pneumoniae",
        "proteus mirabilis",
        "klebsiella oxytoca",
    ]

    # Keep only relevant columns that actually exist
    existing_cols = [col for col in antibiogram_cols if col in mcs_df.columns]

    df = mcs_df[existing_cols].copy()

    # Standardise organism names
    df["organism_bug"] = df["organism_bug"].str.lower().str.strip()

    # Keep only target organisms
    df = df[df["organism_bug"].isin(target_organisms)].copy()

    # Combine co-amoxiclav and augmentin
    if "augmentin" in df.columns:
        if "co-amoxiclav" in df.columns:
            df["co-amoxiclav"] = df["co-amoxiclav"].fillna(df["augmentin"])
        else:
            df["co-amoxiclav"] = df["augmentin"]

        df = df.drop(columns="augmentin")

    # Antibiotic columns after cleaning
    id_cols = ["lab_test_id", "organism_bug", "site_grouped"]
    antibiotic_cols = [col for col in df.columns if col not in id_cols]

    # Collapse duplicate rows per sample / organism / site
    grouped = (
        df
        .groupby(id_cols, dropna=False)[antibiotic_cols]
        .agg(lambda x: set(x.dropna()))
    )

    # Remove rows where any antibiotic has conflicting results
    no_conflicts = grouped.map(lambda x: len(x) <= 1)

    grouped = grouped[no_conflicts.all(axis=1)]

    # Convert one-item sets back to values; empty sets become NA
    cleaned = grouped.map(
        lambda x: next(iter(x)) if len(x) == 1 else pd.NA
    )

    cleaned = cleaned.reset_index()

    return cleaned
    
    

def parse_mic(x):
    if pd.isna(x):
        return None, None
    
    x = str(x).strip().lower()
    
    match = re.match(r"^(<=|>=|<|>)?\s*([0-9]*\.?[0-9]+)$", x) #captures optional operator and numeric value in 2 groups
    
    if not match:
        return None, None
    
    operator = match.group(1) or "="
    value = float(match.group(2))
    
    return operator, value


def classify_mic_result(x, s_breakpoint, r_breakpoint):
    if pd.isna(x):
        return np.nan
    
    # Keep already-standardised categorical results
    x_str = str(x).strip().lower()
    
    if "resistant" in x:
        return "R"
    
    if "intermediate" in x:
        return "I"
    
    if "susceptible" in x or "sensitive" in x:
        return "S"
    
    if "optimised dosing" in x:
        return "I"  
    
    op, mic_value = parse_mic(x)
    
    if mic_value is None:
        return np.nan
    
    # Exact or upper-bound MICs
    if op in ["=", "<=", "<"]:
        if mic_value <= s_breakpoint:
            return "S"
        elif mic_value > r_breakpoint:
            return "R"
        else:
            return "I"
    
    # Lower-bound MICs
    if op in [">", ">="]:
        if mic_value > r_breakpoint:
            return "R"
        elif mic_value <= s_breakpoint:
            # This is ambiguous: e.g. >0.25 when S <=0.25
            return np.nan
        else:
            return "I"
    
    return np.nan


def map_mics_to_susceptibility(df):

    mic_breakpoints = {
    "esbl markers (ss = present)":(0,0),
    "amikacin": (8, 8),
    "amoxicillin": (8, 8),
    "augmentin": (8, 8),  # amoxicillin-clavulanate (simplified)
    "cefepime": (1, 4),
    "cefotaxime": (1, 2),
    "cefoxitin": (8, 8),
    "cefpodoxime": (1, 1),  # UTI/oral only
    "ceftazidime": (1, 4),
    "ceftriaxone": (1, 2),
    "cefuroxime": (8, 8),  # oral/IV varies; simplified
    "cephalexin": (8, 8),  # oral only
    "ciprofloxacin": (0.25, 0.5),
    "co-amoxiclav": (8, 8),  # same as augmentin
    "cotrimoxazole": (2, 4),
    "ertapenem": (0.5, 0.5),
    "esbl_markers (ss = present)": None,  # not an MIC → keep as None
    "gentamicin": (2, 2),
    "imipenem": (2, 4),
    "meropenem": (2, 8),
    "nitrofurantoin": (64, 64),  # E. coli UTI only
    "piperacillin-tazobactam": (8, 8),
    "tazocin": (8, 8),  # synonym
    "temocillin": (0.001, 16),  # UTI only
    "tigecycline": (0.5, 0.5),
}
    
    df = df.copy()
    
    for ab, breakpoints in mic_breakpoints.items():
        if breakpoints is None:
            continue
            
        s_breakpoint, r_breakpoint = breakpoints
        
        if ab in df.columns:
            df[ab] = df[ab].apply(
                lambda x: classify_mic_result(x, s_breakpoint, r_breakpoint)
            )
    
    return df

def get_antibiogram_plot(
    antibiogram_df_clean,
    ordered_abx=None,
    plot_title = "Antibiogram heatmap: % susceptible by organism and antibiotic",
    figsize=(20, 6),
    cmap="RdYlGn",
    save_path=None, 
    ax = None 
):
    """
    Generate an antibiogram heatmap showing % susceptible
    by organism and antibiotic.

    Parameters
    ----------
    antibiogram_df_clean : pd.DataFrame
        DataFrame containing:
            - organism_bug
            - antibiotic result columns with values like "S", "R", etc.

    ordered_abx : list, optional
        List specifying antibiotic column order.

    figsize : tuple, default=(20, 6)
        Figure size.

    cmap : str, default="RdYlGn"
        Heatmap colour map.

    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    heatmap_df : pd.DataFrame
        Pivot table used for plotting (% susceptible).

    n_tested_df : pd.DataFrame
        Pivot table of number tested.

    fig, ax
        Matplotlib figure and axis.
    """


    # -----------------------------
    # Convert to long format
    # -----------------------------
    df_long = antibiogram_df_clean.melt(
        id_vars="organism_bug",
        var_name="antibiotic",
        value_name="result"
    ).dropna(subset=["result"])

    # -----------------------------
    # % susceptible
    # -----------------------------
    pct_s = (
        df_long
        .groupby(["organism_bug", "antibiotic"])["result"]
        .apply(lambda x: (x == "S").mean() * 100)
        .reset_index(name="pct_susceptible")
    )

    # -----------------------------
    # Number tested
    # -----------------------------
    n_tested = (
        df_long
        .groupby(["organism_bug", "antibiotic"])
        .size()
        .reset_index(name="n_tested")
    )

    # -----------------------------
    # Pivot tables
    # -----------------------------
    heatmap_df = pct_s.pivot(
        index="organism_bug",
        columns="antibiotic",
        values="pct_susceptible"
    )

    n_tested_df = n_tested.pivot(
        index="organism_bug",
        columns="antibiotic",
        values="n_tested"
    )

    # Optional antibiotic ordering
    if ordered_abx is not None:
    
        # Add missing columns to BOTH dfs
        cols_to_add_heatmap = [c for c in ordered_abx if c not in heatmap_df.columns]
        cols_to_add_tested = [c for c in ordered_abx if c not in n_tested_df.columns]
    
        for c in cols_to_add_heatmap:
            heatmap_df[c] = np.nan
    
        for c in cols_to_add_tested:
            n_tested_df[c] = np.nan
    
        # Reorder columns safely
        heatmap_df = heatmap_df[ordered_abx]
        n_tested_df = n_tested_df[ordered_abx]

    # -----------------------------
    # Plot heatmap
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_df,
        annot=False,
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "% susceptible"},
        ax=ax
    )

    # -----------------------------
    # Add annotations
    # -----------------------------
    for y, organism in enumerate(heatmap_df.index):

        for x, antibiotic in enumerate(heatmap_df.columns):

            pct = heatmap_df.loc[organism, antibiotic]
            n = n_tested_df.loc[organism, antibiotic]

            # Skip missing cells
            if pd.isna(pct):
                continue

            # Main percentage
            ax.text(
                x + 0.5,
                y + 0.40,
                f"{pct:.0f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black"
            )

            # Smaller n underneath
            ax.text(
                x + 0.5,
                y + 0.72,
                f"(n={int(n)})",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    # -----------------------------
    # Add organism sample counts
    # -----------------------------
    sample_counts = antibiogram_df_clean["organism_bug"].value_counts()

    new_labels = [
        f"{org}\n(n={sample_counts[org]})"
        for org in heatmap_df.index
    ]

    ax.set_yticklabels(new_labels, rotation=0)

    # -----------------------------
    # Labels and formatting
    # -----------------------------
    plt.title(plot_title)

    plt.xlabel("Antibiotic")
    plt.ylabel("Organism", labelpad=20)

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # -----------------------------
    # Save figure
    # -----------------------------
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return heatmap_df, n_tested_df, fig, ax



    heatmap_df, n_tested_df, fig, ax = get_antibiogram(
    antibiogram_df_clean=antibiogram_df,
    ordered_abx=ordered_abx,
    save_path="antibiogram_heatmap.png"
)

    