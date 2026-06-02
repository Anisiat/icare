import numpy as np 
import pandas as pd
import logging
from pathlib import Path
import sys

from src.utils import data_cleaning_tools as dct
from src.utils import micro_cleaning_tools as mct
from src.utils import eda_tools as eda


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")


def clean_mapping_df(df, 
                    cols_to_keep = ['code', 'main_specialty', 'treatment_function' ],
                    rename_col_dict = { 'main specialty title':'main_specialty',
                                        'treatment function title': 'treatment_function' }, 
                                        ):

    df = dct.clean_col_names(df)

    df.rename(columns= rename_col_dict , inplace= True)
    

    #drop unwanted and strip and lower col values 
    for col in df.columns:
        if col not in cols_to_keep:
            df.drop(columns=col, inplace = True)
        else:
            df[col] = df[col].str.lower().str.strip()

    return df


def convert_df_to_dict(map_df, value_col, key_col = 'code'):

    map_dict = map_df.set_index(key_col)[value_col].to_dict()

    #add missing maping key 
    map_dict['326'] = 'acute internal medicine'

    map_dict = { k: v for k, v in map_dict.items() if k.isdigit()}

    return map_dict


def map_codes(df, mapping_dict, col, new_col):

    df[col] = df[col].astype('string').str.strip()
    df[new_col] = df[col].map(mapping_dict)

    # check if all codes are mapped 
    map_codes = list(mapping_dict.keys())
    df_codes = df[col].unique()

    print(f"Code(s) not mapped: {[code for code in df_codes if code not in map_codes]}")

    return df


def resolve_conflict_surgeries(df, conflict_col, group_by_col = 'lab_test_id'):

    df = df.copy()

    df[conflict_col] = df[conflict_col].astype('string')

    not_surgeries = ['clinical haematology', 'gastroenterology', 'endocrinology', 'blood and marrow transplantation', 'paediatric clinical haematology', 'geriatric medicine', 'general medicine']

    not_surgery_mask = (
        df[conflict_col].isin(not_surgeries) &
        df['main_specialty'].str.contains('surgery', case = False, na = False))

    df.loc[not_surgery_mask, conflict_col] = df.loc[not_surgery_mask, 'main_specialty']

    #get global freq table 

    freq = df[conflict_col].value_counts().to_dict()
    # aggregate per lab sample

    aggregated_tfc_df = (
        df.groupby(group_by_col)[[conflict_col, 'procedure_desc']]
        .agg(set)
        .reset_index()
        )

    # find samples with conflicting values
    conflict_mask = aggregated_tfc_df[conflict_col].apply(lambda s: len(s) > 1)

    def choose_surgery_code(desc_set):

        desc_set = list(desc_set)
        desc_set = [d for d in desc_set if d not in not_surgeries]
        
        surgery_desc = [desc for desc in desc_set if 'surgery' in desc.lower()]

        if len(surgery_desc) == 1:
            
            return surgery_desc[0]

        else:
            return max(desc_set, key = lambda v: freq.get(v,0))

    # resolve conflicts 
    aggregated_tfc_df.loc[conflict_mask, conflict_col] = (
        aggregated_tfc_df.loc[conflict_mask, conflict_col]
        .apply(choose_surgery_code)
    )

    #flatten from set to string
    aggregated_tfc_df[conflict_col] = (
        aggregated_tfc_df[conflict_col]
        .apply(lambda x: next(iter(x)) if isinstance(x, set) and len(x) == 1 else x)
    )

    # merge back into main df
    df = df.drop( columns = [conflict_col]).merge(
        aggregated_tfc_df[[group_by_col, conflict_col]],
        on=group_by_col,
        how = 'left'
    )

    df = df[~df[conflict_col].isin(not_surgeries)]

    return df

    
