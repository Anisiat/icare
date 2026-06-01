import pandas as pd
import numpy as np

def get_crp(main_df, cfg_path, join_cols=["subject", 'admission_date'], micro_time = 'first_culture_date', crp_time = 'crp_performed_dt', crp_value_col = 'crp_value'):
    """
    For each patient, keep the CRP measured closest before first_culture_date, then return one row per patient in crp column.
    
    Parameters
    ----------
    main_df : pd.DataFrame
        Main dataframe containing patient rows and first_culture_date.
    join_cols : str or list[str]
        Columns to join on.
    micro_time : str
        Column name for the time of the microbiology test.
    crp_time : str
        Column name for the time of the CRP measurement.
    crp_value_col : str
        Column name for the CRP value.
    Returns 
    -------
    pd.DataFrame
        Main dataframe with one row per patient and a crp column containing the CRP measurement closest to first_culture_date.
    """

    #Load CRP dataframe
    cfg = dct.load_config()
    crp_path = cfg['paths']['crp_path']
    crp_df = pd.read_csv(crp_path)

    crp_df = dct.clean_df_columns(crp_df)

    # Make sure join_cols is a list
    if isinstance(join_cols, str):
        join_cols = [join_cols]

    # Keep only needed columns from main_df
    micro_times = main_df[join_cols + [micro_time]].copy()

    # Merge microbiology time onto CRP
    merged_df = micro_times.merge(crp_df, on=join_cols, how="left")

    # Convert times to datetime
    merged_df[micro_time] = pd.to_datetime(merged_df[micro_time])
    merged_df[crp_time] = pd.to_datetime(merged_df[crp_time].dt.date) #keep only date part of crp_time to avoid issues with multiple measurements on same day

    # Calculate time difference between CRP measurement and micro time
    merged_df['time_diff'] = (merged_df[micro_time] - merged_df[crp_time]).abs()

    # Keep the CRP measurement closest to micro time for each patient
    closest_crp = merged_df.loc[merged_df.groupby(join_cols)['time_diff'].idxmin()]

    # Keep only relevant columns
    result_df = closest_crp[join_cols + [crp_value_col]].copy()
    result_df.rename(columns={crp_value_col: 'crp'}, inplace=True)

    result_df = closest_crp[join_cols + [crp_value
_col]].copy()   
def get_temperature(main_df, cfg_path, join_cols=["subject", 'admission_date'], micro_time = 'first_culture_date', temp_time = 'temp_performed_dt', temp_value_col = 'temperature'):
    """
    For each patient, keep the temperature measured closest before first_culture_date, then return one row per patient in temp column.
    
    Parameters
    ----------
    main_df : pd.DataFrame
        Main dataframe containing patient rows and first_culture_date.
    join_cols : str or list[str]
        Columns to join on.
    micro_time : str
        Column name for the time of the microbiology test.
    temp_time : str
        Column name for the time of the temperature measurement.
    temp_value_col : str
        Column name for the temperature value.
    Returns 
    -------
    pd.DataFrame
        Main dataframe with one row per patient and a temp column containing the temperature measurement closest to first_culture_date.
    """

    #Load temperature dataframe
    cfg = dct.load_config()
    temp_path = cfg['paths']['temperature_path']
    temp_df = pd.read_csv(temp_path)

    temp_df = dct.clean_df_columns(temp_df)

    # Make sure join_cols is a list
    if isinstance(join_cols, str):
        join_cols = [join_cols]

    # Keep only needed columns from main_df
    micro_times = main_df[join_cols + [micro_time]].copy()

    # Merge microbiology time onto temperature
    merged_df = micro_times.merge(temp_df, on=join_cols, how="left")

    # Convert times to datetime
    merged_df[micro_time] = pd.to_datetime(merged_df[micro_time])
    merged_df[temp_time] = pd.to_datetime(merged_df[temp_time].dt.date) #keep only date part of temp_time to avoid issues with multiple measurements on same day

    # Calculate time difference between temp measurement and micro time
    merged_df['time_diff'] = (merged_df[micro_time] - merged_df[temp_time]).abs()

    # Keep the temp measurement closest to micro time for each patient
    closest_temp = merged_df.loc[merged_df.groupby(join_cols)['time_diff'].idxmin()]

    # Keep only relevant columns
    result_df = closest_temp[join_cols + [temp_value_col]].copy()
    result_df.rename(columns={temp_value_col: 'temp'}, inplace=True)

    result_df = closest_temp[join_cols + [temp_value_col]].copy()

    result_df.rename(
        columns={temp_value_col: 'temp'},
        inplace=True
    )

    main_df = main_df.merge(
        result_df,
        on=join_cols,
        how='left'
    )

    return main_df
