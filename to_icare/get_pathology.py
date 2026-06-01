import pandas as pd
import numpy as np

def get_pathology(main_df,
                   pathology_df, 
                   join_cols,
                   micro_time = 'first_culture_date', 
                   admission_date = 'admission_date',
                   pathology_time = 'result_available_dt', 
                   result_col = 'pathology_value',
                   upper_bound_col = '',
                   lower_bound_col = '',
                   test_codes = ['crp'],
                   test_code_col = 'test_code'):
    """
    For each patient and each pathology result type, keep the result measured
    closest before first_culture_date, then return one row per patient
    with one column per pathology result.

    Parameters
    ----------
    main_df : pd.DataFrame
        Main dataframe containing patient rows and first_culture_date.
    pathology_df : pd.DataFrame
        Long-format pathology dataframe containing:
        - join columns
        - test_code
        - pathology_time
        - pathology_value
        - upper_bound 
        - lower_bound 
    join_cols : str or list[str]
        Column(s) to join on.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with one row per patient and one column per pathology result + column that flags if the result
        is above or below normal range.
    """

    # Make sure join_cols is a list
    if isinstance(join_cols, str):
        join_cols = [join_cols]

    if isinstance(test_codes, str):
        test_codes = [test_codes]

    # Keep only needed columns from main_df
    micro_times = main_df[join_cols + [micro_time, admission_date]].copy()

    # Merge microbiology time onto pathology
    merged_df = micro_times.merge(pathology_df, on=join_cols, how="left")

    # Keep only the requested pathology test codes
    if test_codes:
        merged_df = merged_df.loc[merged_df[test_code_col].isin(test_codes)].copy()

    # Convert times to datetime
    merged_df[micro_time] = pd.to_datetime(merged_df[micro_time])
    merged_df[pathology_time] = pd.to_datetime(merged_df[pathology_time])
    merged_df[admission_date] = pd.to_datetime(merged_df[admission_date])

    # Time difference: culture time - pathology time
    merged_df["delta_time"] = merged_df[micro_time] - merged_df[pathology_time]
    merged_df['after_admission'] = merged_df[pathology_time] >= merged_df[admission_date]

    # Keep only pathology results before or exactly at culture time but after admission
    merged_df = merged_df.loc[
        merged_df["delta_time"].notna() & (merged_df["delta_time"] >= pd.Timedelta(0) & (merged_df['after_admission'] == True))
    ].copy()

    # Sort so the closest prior pathology result comes first
    merged_df = merged_df.sort_values(join_cols + [test_code_col, "delta_time"])

    # Keep the closest prior value for each patient and each pathology type
    closest_pathology = merged_df.drop_duplicates(
        subset=join_cols + [test_code_col],
        keep="first"
    )

    #flag if result is above or below normal range
    if upper_bound_col and lower_bound_col:
        closest_pathology['path_out_of_range'] = np.where(
            closest_pathology[result_col].notna(),
            ((closest_pathology[result_col] > closest_pathology[upper_bound_col]) |
             (closest_pathology[result_col] < closest_pathology[lower_bound_col])).astype(int),
            np.nan,
        )

    # Pivot to wide format: one column per pathology result and flag
    values_to_pivot = [result_col]
    if upper_bound_col and lower_bound_col:
        values_to_pivot.append('path_out_of_range')

    pathology_wide = closest_pathology.pivot(
        index=join_cols + [admission_date],
        columns=test_code_col,
        values=values_to_pivot
    ).reset_index()

    # Optional: flatten column index name
    pathology_wide.columns = [
        col if isinstance(col, str) else f"{col[1]}_{col[0]}"
        for col in pathology_wide.columns.to_flat_index()
    ]

    # Make the wide result columns a little more readable
    rename_map = {}
    for code in test_codes:
        rename_map[f"{code}_{result_col}"] = code
        if upper_bound_col and lower_bound_col:
            rename_map[f"{code}_path_out_of_range"] = f"{code}_out_of_range"
    pathology_wide = pathology_wide.rename(columns=rename_map)

    return pathology_wide

if __name__ == "__main__":
    main_df = pd.read_csv("data/main_df.csv")
    pathology_df = pd.read_csv("data/pathology_df.csv")
    join_cols = "patient_id"
    pathology_wide = get_pathology(main_df, pathology_df, join_cols)
    pathology_wide.to_csv("data/pathology_wide.csv", index=False)
