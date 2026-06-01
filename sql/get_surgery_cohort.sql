-- =============================================================================
-- ELECTIVE SURGERY COHORT + MICROBIOLOGY LINKAGE (EKP)
--
-- PURPOSE
--   1. Identify adult elective surgery admissions.
--   2. Exclude emergency procedures.
--   3. Deduplicate surgery records.
--   4. Link surgery patients to the microbiology dataset.
--   5. Run QA checks and summary counts.
-- =============================================================================


-- =============================================================================
-- STEP 1: BUILD ELECTIVE SURGERY COHORT
-- =============================================================================

CREATE OR REPLACE TABLE AT_ELECTIVE_SURGERY_COHORT AS (

WITH all_surgeries AS (

    SELECT
        s.SUBJECT,
        s.ENCNTR_ID,
        s.PROCEDURE_DESC,
        s.SURGERY_START_DT,
        s.SURGERY_STOP_DT,
        s.SURGICAL_AREA,
        s.THEATRE_NBR,

       e.ADMISSION_DATE,
        e.ADMISSION_METHOD,
        e.ADMISSION_METHOD_DESC,
        e.ADMISSION_SOURCE,
        e.ADMISSION_SOURCE_DESC,
        e.ADMISSION_TIME,
        e.AGE_AT_ADMISSION,
        e.DISCHARGE_DATE,
        e.DISCHARGE_DESTINATION,
        e.DISCHARGE_DESTINATION_DESC,
        e.DISCHARGE_METHOD,
        e.DISCHARGE_METHOD_DESC,
        e.DISCHARGE_TIME,
        e.EPISODE_END_DATE,
        e.EPISODE_END_TIME,
        e.EPISODE_IDENTIFIER,
        e.EPISODE_START_DATE,
        e.EPISODE_START_TIME,
        e.INDEX_OF_MULTIPLE_DEPRIVATION_DECILE,
        e.MAIN_SPECIALTY_CODE,
        e.MAIN_SPECIALTY_CODE_DESC,
        e.ORDER_NO_OF_EPISODE,
        e.POSTCODE_ON_ADMISSION,
        e.SPELL_IDENTIFIER,
        e.TREATMENT_FUNCTION_CODE,
        e.TREATMENT_FUNCTION_CODE_DESC

    FROM ICHT_PROD.ICARE_ICHT.ICARE_SURGERY_ANON s

    INNER JOIN ICHT_PROD.ICARE_ICHT.ICARE_EPISODES_ANON e
        ON s.SUBJECT = e.SUBJECT
       AND s.SURGERY_STOP_DT >= e.ADMISSION_DATE
       AND s.SURGERY_STOP_DT < DATEADD(day, 1, e.DISCHARGE_DATE)

    WHERE e.DISCHARGE_DATE != 'none'
      AND e.ADMISSION_METHOD IN ('11', '12', '13')
      AND e.AGE_AT_ADMISSION >= 18
      AND s.SURGERY_STOP_DT <= '2025-08-31'
),


-- -----------------------------------------------------------------------------
-- Flag surgery events containing any procedure labelled as emergency.
-- The MAX() window means that if ANY procedure within the surgery event is
-- emergency-related, all rows for that surgery receive has_emergency_flag = 1.
-- -----------------------------------------------------------------------------
flagged AS (

    SELECT
        *,
        MAX(
            CASE
                WHEN LOWER(PROCEDURE_DESC) LIKE '%emergency%' THEN 1
                ELSE 0
            END
        ) OVER (
            PARTITION BY SUBJECT,
                         SPELL_IDENTIFIER,
                         SURGERY_START_DT,
                         SURGERY_STOP_DT
        ) AS has_emergency_flag

    FROM all_surgeries
),


-- Keep only non-emergency surgeries
non_emergencies AS (

    SELECT *
    FROM flagged
    WHERE has_emergency_flag = 0
),


-- -----------------------------------------------------------------------------
-- Deduplicate surgery rows.
-- Duplicate definition is based on:
--   SUBJECT
--   SPELL_IDENTIFIER
--   SURGERY_START_DT
--   SURGERY_STOP_DT
--   PROCEDURE_DESC
--   TREATMENT_FUNCTION_CODE
-- -----------------------------------------------------------------------------
deduplicated AS (

    SELECT *
    FROM non_emergencies

    QUALIFY ROW_NUMBER() OVER (

        -- choose which columns to deduplicate by
        PARTITION BY
            SUBJECT,
            SPELL_IDENTIFIER,
            SURGERY_START_DT,
            SURGERY_STOP_DT,
            PROCEDURE_DESC,
            TREATMENT_FUNCTION_CODE

        ORDER BY SPELL_IDENTIFIER

    ) = 1
)

SELECT *
FROM deduplicated

);




-- =============================================================================
-- STEP 2: MERGE SURGERY COHORT WITH MICROBIOLOGY DATASET
-- =============================================================================

CREATE OR REPLACE TABLE AT_ELECTIVE_SURGERY_MICRO_EKP AS (

SELECT
    m.*,

    s.SUBJECT,
    s.ENCNTR_ID,
    s.PROCEDURE_DESC,
    s.SURGERY_START_DT,
    s.SURGERY_STOP_DT,
    s.SURGICAL_AREA,
    s.THEATRE_NBR,
    s.ADMISSION_DATE,
    s.ADMISSION_METHOD,
    s.ADMISSION_METHOD_DESC,
    s.ADMISSION_SOURCE,
    s.ADMISSION_SOURCE_DESC,
    s.ADMISSION_TIME,
    s.AGE_AT_ADMISSION,
    s.DISCHARGE_DATE,
    s.DISCHARGE_DESTINATION,
    s.DISCHARGE_DESTINATION_DESC,
    s.DISCHARGE_METHOD,
    s.DISCHARGE_METHOD_DESC,
    s.DISCHARGE_TIME,
    s.EPISODE_END_DATE,
    s.EPISODE_END_TIME,
    s.EPISODE_IDENTIFIER,
    s.EPISODE_START_DATE,
    s.EPISODE_START_TIME,
    s.INDEX_OF_MULTIPLE_DEPRIVATION_DECILE,
    s.MAIN_SPECIALTY_CODE,
    s.MAIN_SPECIALTY_CODE_DESC,
    s.ORDER_NO_OF_EPISODE,
    s.POSTCODE_ON_ADMISSION,
    s.SPELL_IDENTIFIER,
    s.TREATMENT_FUNCTION_CODE,
    s.TREATMENT_FUNCTION_CODE_DESC

FROM AT_ELECTIVE_SURGERY_COHORT s

INNER JOIN AT_MICROBIOLOGY_EKP m
    ON s.SUBJECT = m.SUBJECT

);



-- =============================================================================
-- QA CHECKS / DATASET CHARACTERISATION
-- =============================================================================

-- Distinct lab tests, subjects and total linked rows.
-- Screenshot comment indicates expected values:
-- 19667 lab_ids, 8579 subjects, 42829 rows.
SELECT
    COUNT(DISTINCT LAB_TEST_ID) AS lab_ids,
    COUNT(DISTINCT SUBJECT) AS subjects,
    COUNT(*) AS total
FROM AT_ELECTIVE_SURGERY_MICRO_EKP; -- 19667, 8579, 42829



-- Overall dataset summary
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT SUBJECT) AS n_patients,
    COUNT(DISTINCT SPELL_IDENTIFIER) AS n_spells,
    COUNT(DISTINCT LAB_TEST_ID) AS n_lab_tests,
    COUNT(
        DISTINCT SUBJECT || '|' ||
        SPELL_IDENTIFIER || '|' ||
        SURGERY_START_DT || '|' ||
        SURGERY_STOP_DT
    ) AS n_surgeries
FROM AT_ELECTIVE_SURGERY_MICRO_EKP;


-- Identify duplicate microbiology rows within a surgery event
SELECT
    SUBJECT,
    SPELL_IDENTIFIER,
    SURGERY_START_DT,
    SURGERY_STOP_DT,
    LAB_TEST_ID,
    COUNT(*) AS n_rows
FROM AT_ELECTIVE_SURGERY_MICRO_EKP
GROUP BY
    SUBJECT,
    SPELL_IDENTIFIER,
    SURGERY_START_DT,
    SURGERY_STOP_DT,
    LAB_TEST_ID
HAVING COUNT(*) > 1
ORDER BY n_rows DESC;


-- Count frequencies of lab_test_id values
-- (comment says "count values for surgery types", but the code groups lab_test_id)
SELECT
    LOWER(lab_test_id) AS lab_test_id,
    COUNT(*) AS n
FROM AT_ELECTIVE_SURGERY_MICRO_EKP
GROUP BY LOWER(lab_test_id)
ORDER BY n DESC;


-- Percentage of surgery patients in each hospital (using SURGICAL_AREA)
SELECT
    SURGICAL_AREA AS hospital,
    COUNT(DISTINCT SUBJECT) AS n_surgery_patients,
    ROUND(
        100.0 * COUNT(DISTINCT SUBJECT)
        / SUM(COUNT(DISTINCT SUBJECT)) OVER (),
        2
    ) AS pct_surgery_patients
FROM AT_ELECTIVE_SURGERY_MICRO_EKP
GROUP BY SURGICAL_AREA
ORDER BY pct_surgery_patients DESC, hospital;


-- =============================================================================
-- REQUESTED COUNTS: TOTAL ELECTIVE SURGERY PATIENTS, ANY INFECTION, ESBL
-- =============================================================================
-- Assumption: AT_MICROBIOLOGY_EKP (and therefore AT_ELECTIVE_SURGERY_MICRO_EKP)
-- contains positive microbiology cultures.
SELECT
    (SELECT COUNT(DISTINCT SUBJECT, SURGERY_START_DT, SURGERY_STOP_DT)
     FROM AT_ELECTIVE_SURGERY_COHORT) AS total_elective_surgeries, 

    (SELECT COUNT(DISTINCT SUBJECT)
     FROM AT_ELECTIVE_SURGERY_COHORT) as total_elective_surgery_patients,

    (SELECT COUNT(DISTINCT SUBJECT)
     FROM AT_ELECTIVE_SURGERY_MICRO_EKP) AS elective_surgery_patients_with_positive_ekp
     
     
    -- get average LOS for patients with and without infection (defined as having a linked microbiology record)
    
    (Select ROUND(AVG(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)), 2) AS avg_los_with_infection
     FROM AT_ELECTIVE_SURGERY_MICRO_EKP),
     
     (Select ROUND(AVG(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)), 2) AS avg_los_without_infection
     FROM AT_ELECTIVE_SURGERY_COHORT
     WHERE SUBJECT NOT IN (SELECT DISTINCT SUBJECT FROM AT_ELECTIVE_SURGERY_MICRO_EKP));



