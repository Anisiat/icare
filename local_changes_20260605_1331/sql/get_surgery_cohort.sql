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
-- keeps microbiology samples linked to surgery patients during the whole 10y window
-- 48h post admission +  surgical spell filter applied later in python)
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


