import numpy as np 
import pandas as pd
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
INTERIM_DIR = DATA_DIR / 'interim'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import data_cleaning_tools as dct

asthma_drugs = [
    'salbutamol',
    'beclometasone',
    'budesonide',
    'fluticasone',
    'beclometasone-formoterol',
    'budesonide-formoterol',
    'fluticasone-salmeterol',
    'fluticasone-vilanterol',
    'fluticasone-formoterol'
]

copd_drugs = [
    'tiotropium',
    'glycopyrronium',
    'aclidinium',
    'umeclidinium',
    'glycopyrronium-indacaterol',
    'umeclidinium-vilanterol',
    'fluticasone/umeclidinium/vilanterol',
    'beclometasone/formoterol/glycopyrronium',
    'indacaterol',
    'salmeterol'
]

hypertension_drugs = [
    'amlodipine', 'nifedipine', 'felodipine',   # CCBs
    'lisinopril', 'perindopril',                # ACE inhibitors
    'losartan', 'irbesartan', 'candesartan',    # ARBs
    'bisoprolol', 'atenolol', 'metoprolol',     # beta-blockers
    'indapamide', 'bendroflumethiazide',        # thiazides
    'doxazosin'                                 # alpha-blocker
]

ihd_drugs = [
    'glyceryl trinitrate', 'isosorbide mononitrate',  # nitrates
    'clopidogrel',                         # antiplatelets (if present)
    'atorvastatin', 'simvastatin',                    # statins (if present)
    'ranolazine'                                      # angina-specific
]

type2_diabetes_drugs = [
    'metformin',
    'gliclazide',
    'linagliptin',
    'sitagliptin',
    'alogliptin',
    'dapagliflozin',
    'empagliflozin',
    'canagliflozin',
    'semaglutide',
    'liraglutide',
    'dulaglutide'
]

ckd_drugs = [
    'sodium bicarbonate'   # metabolic acidosis in CKD
]

comorb_dict = {
    'asthma': asthma_drugs,
    'copd': copd_drugs,
    'hypertension': hypertension_drugs,
    'ischaemic_heart_disease': ihd_drugs,
    'renal_failure': ckd_drugs,
    'type2_diabetes': type2_diabetes_drugs
}


def infer_comorb(
    prescriptions_df,
    medication_col = 'medication_name_short',
    comorb_dict = comorb_dict,
    subject_col='subject',
    admission_col='admission_date'
):
    subject_medications = (
        prescriptions_df
        .groupby([subject_col, admission_col])[medication_col]
        .agg(list)
        .rename('medications')
        .reset_index()
    )

    subject_medications['medications'] = subject_medications['medications'].apply(
        lambda meds: [str(m).strip().lower() for m in meds]
    )

    for comorb, meds in comorb_dict.items():
        meds_set = {str(m).strip().lower() for m in meds}
        subject_medications[comorb] = subject_medications['medications'].apply(
            lambda patient_meds: int(any(med in meds_set for med in patient_meds))
        )

    return subject_medications


def update_comorbidities(
    main_df,
    cfg_path,
    comorb_cols = list(comorb_dict.keys()),
    join_cols = ['subject', 'admission_date'],
    return_update_summary = True
):

    cfg = dct.load_config(cfg_path)
    paths = cfg.get('paths', {})

    # Load prescriptions
    prescriptions_path = PROJECT_ROOT / paths['prescriptions']
    prescriptions_df = pd.read_csv(prescriptions_path)
    prescriptions_df = dct.clean_df_columns(prescriptions_df)
    
    inferred_df = infer_comorb(prescriptions_df)
    inferred_subset = inferred_df[join_cols + comorb_cols].copy()

    df = main_df.merge(
        inferred_subset,
        on=join_cols,
        how='left',
        suffixes=('', '_inferred')
    )

    summary = {}
    total_new = 0

    for col in comorb_cols:
        inferred_col = f"{col}_inferred"

        original = df[col].fillna(0).astype(int)
        inferred = df[inferred_col].fillna(0).astype(int)

        new_cases = ((original == 0) & (inferred == 1)).sum()
        df[col] = original | inferred

        summary[col] = int(new_cases)
        total_new += new_cases

    df = df.drop(columns=[f"{col}_inferred" for col in comorb_cols])
    summary["total_new_comorbidities"] = int(total_new)

    if return_update_summary:
        print(summary)
        
    return df