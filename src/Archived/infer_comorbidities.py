
## get comorbidities dictionary 

## ----------------------------------------------------------

comorbidities = ['anemia','asthma', 'cancer', 'copd', 'hypertension', 'ischaemic_heart_disease',
       'obesity', 'renal_failure', 'type2_diabetes']

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
    'aspirin', 'clopidogrel',                         # antiplatelets (if present)
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

## ----------------------------------------------------------

def infer_comorb(prescriptions_df, medication_col, comorb_dict, subject_col='subject', admission_col='admission_date'):
    subject_medications = (
        prescriptions_df
        .groupby([subject_col, admission_col])[medication_col]
        .agg(list)
        .rename('medications')
        .reset_index()
    )

    # Normalize medication strings once to make matching case-insensitive.
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
    inferred_df,
    comorb_cols,
    join_cols
):
    """
    Update binary comorbidity columns in main_df using inferred_df.

    Logic:
    - If inferred == 1 → set main_df[col] = 1
    - Otherwise keep original value

    Returns:
    - updated_df
    - summary dict with counts of newly inferred comorbidities
    """

    # Merge inferred data
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

        # Ensure binary ints and fill missing inferred values with 0
        original = df[col].fillna(0).astype(int)
        inferred = df[inferred_col].fillna(0).astype(int)

        # Identify newly added comorbidities (0 → 1)
        new_cases = ((original == 0) & (inferred == 1)).sum()

        # Update using logical OR
        df[col] = original | inferred

        summary[col] = int(new_cases)
        total_new += new_cases

    # Drop temporary inferred columns
    df = df.drop(columns=[f"{col}_inferred" for col in comorb_cols])

    summary["total_new_comorbidities"] = int(total_new)

    return df, summary