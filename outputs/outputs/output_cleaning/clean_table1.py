import numpy as np
import pandas as pd

def import_raw_csv(file_path = '../table1/table1_epidemiology_esbl.csv'):
    """
    Imports a raw CSV file and returns a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The imported data as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error importing CSV file: {e}")
        return None

tfc_mapping = {
    'Abdominal gi': 'Abdominal gastrointestinal',
    'Cardiothoracic': 'Cardiothoracic',
    'General surgery': 'General surgery',
    'Gynaecological oncology': 'Gynaecological oncology',
    'Neuro ent': 'Neurology/ENT',
    'Obs gynae': 'Obstetrics/Gynaecology',
    'Ortho plastics': 'Orthopaedic plastics',
    'Uro nephro': 'Urology/Nephrology'
}

def clean_data(df, tfc_mapping=tfc_mapping):
    """
    Cleans the DataFrame by handling missing values and converting data types.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    # Drop the top weird header row and the sample-size row
    df = df.iloc[2:].copy()

    # Rename columns manually
    df = df.rename(columns={
    "Unnamed: 0": "variable",
    "Unnamed: 1": "level",
    "Grouped by esbl_status": "missing",
    "Grouped by esbl_status.1": "overall",
    "Grouped by esbl_status.2": "ESBL",
    "Grouped by esbl_status.3": "non-ESBL",
    "Grouped by esbl_status.4": "p_value",
    })

    # Reset row numbers
    df = df.reset_index(drop=True)

    # capitalise column names
    for c in df.columns:
        if c.lower() not in ['esbl', 'non-esbl']:
            df[c] = (
                df[c]
                .replace('_', ' ', regex=True)
                .replace('desc', ' ', regex=True)
                .str.capitalize()
                .replace('esbl', 'ESBL', regex=True)
                .replace('non-esbl', 'non-ESBL', regex=True)
                .replace('abx', 'antibiotics', regex=True)
                .replace('Copd', 'COPD', regex=True)
                .replace('Crp', 'CRP', regex=True)
                .replace('hosp', 'hospital', regex=True)
                .replace('Tfc', 'Surgical specialty', regex=True)
            )

    # remove 0s in missing column and replace with NaN
    df['missing'] = df['missing'].replace('0', np.nan)
    
    # map TFC values using the provided mapping
    df['level'] = df['level'].replace(tfc_mapping)

    # remove binary rows and only keep positive rows for binary variables
    idx_to_drop = df[(df['level'] == '0') & (df['variable'] != '')].index
    
    df = df.drop(idx_to_drop).reset_index(drop=True)

    # remove repeated variable names in the "variable" column
    df.loc[df["variable"] == df["variable"].shift(), "variable"] = ""
    
    for v in df['variable']:
        if 'gender' in v.lower():
            df.loc[df['variable'] == v, 'level'] = 'Male'
        else:
            df['level'] = df['level'].replace('1', '')

    return df



def main():
    # Import the raw CSV file
    df_raw = import_raw_csv()

    if df_raw is not None:
        # Clean the data
        df_cleaned = clean_data(df_raw)
    
    print(df_cleaned.head())
    
    df_cleaned.to_csv('../table1/table1_epidemiology_esbl_cleaned.csv', index=False)


if __name__ == "__main__":
    main()