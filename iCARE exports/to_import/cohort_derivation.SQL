/* ============================================================
   COHORT FLOW NUMBERS FOR ELECTIVE SURGERY + EKP MICROBIOLOGY
   Study period: Jan 2015 to Aug 2025 inclusive
   Cutoff used: latest_collect_dt < 2025-09-01
   ============================================================ */


-- ============================================================
-- 1. All microbiology samples in study period
-- ============================================================

SELECT
    '01_all_microbiology_samples' AS flow_step,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests
FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
WHERE latest_collect_dt >= '2015-01-01'
  AND latest_collect_dt <  '2025-09-01';



-- ============================================================
-- 2. EKP-positive microbiology samples
--    EKP = E. coli, K. pneumoniae, K. oxytoca, P. mirabilis
-- ============================================================

SELECT
    '02_ekp_positive_samples' AS flow_step,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests,
    COUNT(DISTINCT SUBJECT) AS n_patients
FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
WHERE latest_collect_dt >= '2015-01-01'
  AND latest_collect_dt <  '2025-09-01'
  AND LOWER(organism_bug) IN (
        'escherichia coli',
        'klebsiella pneumoniae',
        'klebsiella oxytoca',
        'proteus mirabilis'
  );



-- ============================================================
-- 3. EKP samples with susceptibility data available
--    This is NOT yet ESBL-classifiable, only has AST data.
-- ============================================================

WITH ekp_samples AS (
    SELECT DISTINCT LAB_TEST_ID, SUBJECT
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
    WHERE latest_collect_dt >= '2015-01-01'
      AND latest_collect_dt <  '2025-09-01'
      AND LOWER(organism_bug) IN (
            'escherichia coli',
            'klebsiella pneumoniae',
            'klebsiella oxytoca',
            'proteus mirabilis'
      )
),

ekp_with_ast AS (
    SELECT DISTINCT m.LAB_TEST_ID, m.SUBJECT
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON m
    INNER JOIN ekp_samples e
        ON m.LAB_TEST_ID = e.LAB_TEST_ID
    WHERE LOWER(m.order_name) LIKE '%susceptibility%'
      AND m.sensitivity IS NOT NULL
      AND LOWER(m.sensitivity) != 'none'
)

SELECT
    '03_ekp_samples_with_any_ast' AS flow_step,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests,
    COUNT(DISTINCT SUBJECT) AS n_patients
FROM ekp_with_ast;



-- ============================================================
-- 4. EKP samples classifiable for ESBL / 3GC status
--    These are samples with relevant cephalosporin or ESBL marker data.
--    Samples without these data should be "unknown", not assumed non-ESBL.
-- ============================================================

WITH ekp_ast AS (
    SELECT
        SUBJECT,
        LAB_TEST_ID,
        LOWER(organism_bug) AS organism_bug,
        LOWER(TRIM(SPLIT_PART(sensitivity, ' : ', 1))) AS antibiotic_name,
        LOWER(TRIM(SPLIT_PART(sensitivity, ' : ', 2))) AS sensitivity_result
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
    WHERE latest_collect_dt >= '2015-01-01'
      AND latest_collect_dt <  '2025-09-01'
      AND LOWER(order_name) LIKE '%susceptibility%'
      AND LOWER(organism_bug) IN (
            'escherichia coli',
            'klebsiella pneumoniae',
            'klebsiella oxytoca',
            'proteus mirabilis'
      )
),

classified AS (
    SELECT
        SUBJECT,
        LAB_TEST_ID,

        MAX(
            CASE
                WHEN antibiotic_name IN (
                    'cefpodoxime',
                    'cefotaxime',
                    'ceftriaxone',
                    'ceftazidime',
                    'cefuroxime'
                )
                AND sensitivity_result LIKE '%resistant%'
                THEN 1 ELSE 0
            END
        ) AS has_3gc_resistance,

        MAX(
            CASE
                WHEN antibiotic_name IN (
                    'cefpodoxime',
                    'cefotaxime',
                    'ceftriaxone',
                    'ceftazidime',
                    'cefuroxime'
                )
                AND sensitivity_result LIKE '%susceptible%'
                THEN 1 ELSE 0
            END
        ) AS has_3gc_susceptibility,

        MAX(
            CASE
                WHEN antibiotic_name LIKE '%esbl%'
                 AND sensitivity_result LIKE '%positive%'
                THEN 1 ELSE 0
            END
        ) AS has_esbl_positive,

        MAX(
            CASE
                WHEN antibiotic_name LIKE '%esbl%'
                 AND sensitivity_result LIKE '%negative%'
                THEN 1 ELSE 0
            END
        ) AS has_esbl_negative

    FROM ekp_ast
    GROUP BY SUBJECT, LAB_TEST_ID
),

status AS (
    SELECT
        SUBJECT,
        LAB_TEST_ID,
        CASE
            WHEN has_esbl_positive = 1
              OR has_3gc_resistance = 1
            THEN 'ESBL_or_3GC_R'

            WHEN has_esbl_negative = 1
              OR has_3gc_susceptibility = 1
            THEN 'non_ESBL_or_3GC_S'

            ELSE 'unknown_ESBL_status'
        END AS esbl_status
    FROM classified
)

SELECT
    esbl_status,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests,
    COUNT(DISTINCT SUBJECT) AS n_patients
FROM status
GROUP BY esbl_status
ORDER BY esbl_status;



-- ============================================================
-- 5. Elective adult inpatient surgery cohort
-- ============================================================

WITH all_surgeries AS (
    SELECT
        s.SUBJECT,
        s.PROCEDURE_DESC,
        s.SURGERY_START_DT,
        s.SURGERY_STOP_DT,
        s.SURGICAL_AREA,

        e.SPELL_IDENTIFIER,
        e.EPISODE_IDENTIFIER,
        e.ADMISSION_DATE,
        e.ADMISSION_TIME,
        e.ADMISSION_METHOD,
        e.AGE_AT_ADMISSION,
        e.DISCHARGE_DATE,
        e.DISCHARGE_TIME,
        e.DISCHARGE_DESTINATION,
        e.MAIN_SPECIALTY_CODE,
        e.INDEX_OF_MULTIPLE_DEPRIVATION_DECILE,
        e.TREATMENT_FUNCTION_CODE

    FROM ICHT_PROD.ICARE_ICHT.ICARE_SURGERY_ANON s
    INNER JOIN ICHT_PROD.ICARE_ICHT.ICARE_EPISODES_ANON e
        ON s.SUBJECT = e.SUBJECT
       AND s.SURGERY_STOP_DT >= e.ADMISSION_DATE
       AND s.SURGERY_STOP_DT < DATEADD(DAY, 1, e.DISCHARGE_DATE)

    WHERE e.DISCHARGE_DATE != 'none'
      AND e.ADMISSION_METHOD IN ('11', '12', '13')
      AND e.AGE_AT_ADMISSION >= 18
),

flagged AS (
    SELECT
        *,
        MAX(
            CASE
                WHEN LOWER(PROCEDURE_DESC) LIKE '%emergency%' THEN 1
                ELSE 0
            END
        ) OVER (
            PARTITION BY SUBJECT, SPELL_IDENTIFIER, SURGERY_START_DT, SURGERY_STOP_DT
        ) AS has_emergency_flag
    FROM all_surgeries
),

non_emergency AS (
    SELECT *
    FROM flagged
    WHERE has_emergency_flag = 0
),

deduplicated AS (
    SELECT *
    FROM non_emergency
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY SUBJECT, SPELL_IDENTIFIER, SURGERY_START_DT, SURGERY_STOP_DT
        ORDER BY EPISODE_IDENTIFIER
    ) = 1
)

SELECT
    '05_adult_elective_inpatient_surgeries' AS flow_step,
    COUNT(*) AS n_surgery_rows,
    COUNT(DISTINCT SUBJECT) AS n_patients,
    COUNT(DISTINCT SPELL_IDENTIFIER) AS n_spells,
    COUNT(DISTINCT SUBJECT || '|' || SPELL_IDENTIFIER || '|' || SURGERY_START_DT || '|' || SURGERY_STOP_DT) AS n_surgeries
FROM deduplicated;



-- ============================================================
-- 6. Elective surgery patients with any EKP microbiology sample
--    This is broad matching by patient only.
--    You can do the nosocomial timing filter in Python if preferred.
-- ============================================================

WITH surgery_cohort AS (
    SELECT *
    FROM AT_ELECTIVE_SURGERY_COHORT
),

ekp_micro AS (
    SELECT DISTINCT
        SUBJECT,
        LAB_TEST_ID,
        LATEST_COLLECT_DT,
        ORGANISM_BUG
    FROM AT_MICROBIOLOGY_EKP
)

SELECT
    '06_elective_surgery_patients_with_ekp_micro' AS flow_step,
    COUNT(*) AS n_joined_rows,
    COUNT(DISTINCT s.SUBJECT) AS n_patients,
    COUNT(DISTINCT s.SPELL_IDENTIFIER) AS n_spells,
    COUNT(DISTINCT m.LAB_TEST_ID) AS n_lab_tests
FROM surgery_cohort s
INNER JOIN ekp_micro m
    ON s.SUBJECT = m.SUBJECT;



-- ============================================================
-- 7. Optional SQL version of nosocomial EKP definition
--    Use this only if you want SQL to reproduce your final diagram.
--    Definition:
--    culture collected >=48h after admission
--    and <30 days after discharge
-- ============================================================

WITH surgery_micro AS (
    SELECT
        s.SUBJECT,
        s.SPELL_IDENTIFIER,
        s.SURGERY_START_DT,
        s.SURGERY_STOP_DT,
        s.ADMISSION_DATE,
        s.DISCHARGE_DATE,
        m.LAB_TEST_ID,
        m.LATEST_COLLECT_DT,
        m.ORGANISM_BUG
    FROM AT_ELECTIVE_SURGERY_COHORT s
    INNER JOIN AT_MICROBIOLOGY_EKP m
        ON s.SUBJECT = m.SUBJECT
)

SELECT
    '07_elective_surgery_with_nosocomial_ekp' AS flow_step,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT SUBJECT) AS n_patients,
    COUNT(DISTINCT SPELL_IDENTIFIER) AS n_spells,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests
FROM surgery_micro
WHERE LATEST_COLLECT_DT >= DATEADD(HOUR, 48, ADMISSION_DATE)
  AND LATEST_COLLECT_DT <  DATEADD(DAY, 30, DISCHARGE_DATE);



-- ============================================================
-- 8. Final ESBL / non-ESBL / unknown status among nosocomial EKP
--    This gives final numbers for the bottom of your diagram.
-- ============================================================

WITH nosocomial_ekp AS (
    SELECT
        s.SUBJECT,
        s.SPELL_IDENTIFIER,
        m.LAB_TEST_ID,
        m.LATEST_COLLECT_DT
    FROM AT_ELECTIVE_SURGERY_COHORT s
    INNER JOIN AT_MICROBIOLOGY_EKP m
        ON s.SUBJECT = m.SUBJECT
    WHERE m.LATEST_COLLECT_DT >= DATEADD(HOUR, 48, s.ADMISSION_DATE)
      AND m.LATEST_COLLECT_DT <  DATEADD(DAY, 30, s.DISCHARGE_DATE)
),

ekp_ast AS (
    SELECT
        SUBJECT,
        LAB_TEST_ID,
        LOWER(TRIM(SPLIT_PART(sensitivity, ' : ', 1))) AS antibiotic_name,
        LOWER(TRIM(SPLIT_PART(sensitivity, ' : ', 2))) AS sensitivity_result
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
    WHERE LOWER(order_name) LIKE '%susceptibility%'
      AND LOWER(organism_bug) IN (
            'escherichia coli',
            'klebsiella pneumoniae',
            'klebsiella oxytoca',
            'proteus mirabilis'
      )
),

classified AS (
    SELECT
        n.SUBJECT,
        n.SPELL_IDENTIFIER,
        n.LAB_TEST_ID,

        MAX(
            CASE
                WHEN a.antibiotic_name IN (
                    'cefpodoxime',
                    'cefotaxime',
                    'ceftriaxone',
                    'ceftazidime',
                    'cefuroxime'
                )
                AND a.sensitivity_result LIKE '%resistant%'
                THEN 1 ELSE 0
            END
        ) AS has_3gc_resistance,

        MAX(
            CASE
                WHEN a.antibiotic_name IN (
                    'cefpodoxime',
                    'cefotaxime',
                    'ceftriaxone',
                    'ceftazidime',
                    'cefuroxime'
                )
                AND a.sensitivity_result LIKE '%susceptible%'
                THEN 1 ELSE 0
            END
        ) AS has_3gc_susceptibility,

        MAX(
            CASE
                WHEN a.antibiotic_name LIKE '%esbl%'
                 AND a.sensitivity_result LIKE '%positive%'
                THEN 1 ELSE 0
            END
        ) AS has_esbl_positive,

        MAX(
            CASE
                WHEN a.antibiotic_name LIKE '%esbl%'
                 AND a.sensitivity_result LIKE '%negative%'
                THEN 1 ELSE 0
            END
        ) AS has_esbl_negative

    FROM nosocomial_ekp n
    LEFT JOIN ekp_ast a
        ON n.LAB_TEST_ID = a.LAB_TEST_ID
    GROUP BY
        n.SUBJECT,
        n.SPELL_IDENTIFIER,
        n.LAB_TEST_ID
),

status AS (
    SELECT
        SUBJECT,
        SPELL_IDENTIFIER,
        LAB_TEST_ID,
        CASE
            WHEN has_esbl_positive = 1
              OR has_3gc_resistance = 1
            THEN 'ESBL_or_3GC_R'

            WHEN has_esbl_negative = 1
              OR has_3gc_susceptibility = 1
            THEN 'non_ESBL_or_3GC_S'

            ELSE 'unknown_ESBL_status'
        END AS esbl_status
    FROM classified
)

SELECT
    esbl_status,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests,
    COUNT(DISTINCT SUBJECT) AS n_patients,
    COUNT(DISTINCT SPELL_IDENTIFIER) AS n_spells
FROM status
GROUP BY esbl_status
ORDER BY esbl_status;