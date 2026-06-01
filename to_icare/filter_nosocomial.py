def filter_nosocomial(df, admission_col, sample_collect_dt, discharge_dt):

    """
     Keep only nosocomial infections, defined as those where the sample collection date is at least 48 hours after admission.
    """

    df = df.copy()

    # Convert date columns to datetime if they are not already
    df[admission_col] = pd.to_datetime(df[admission_col])
    df[sample_collect_dt] = pd.to_datetime(df[sample_collect_dt])
    df[discharge_dt] = pd.to_datetime(df[discharge_dt])

    # Get only date from datetime columns
    df[admission_col] = df[admission_col].dt.date
    df[sample_collect_dt] = df[sample_collect_dt].dt.date
    df[discharge_dt] = df[discharge_dt].dt.date

    # Calculate time difference in days between admission and sample collection
    df['time_to_sample_collection'] = (df[sample_collect_dt] - df[admission_col])
    df = df[(df['time_to_sample_collection'] >= 2) & (df[sample_collect_dt] < df[discharge_dt])]

    return df