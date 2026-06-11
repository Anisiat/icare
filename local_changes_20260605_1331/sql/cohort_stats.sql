
-- =============================================================================
-- QA CHECKS / DATASET CHARACTERISATION
-- =============================================================================

-- Distinct lab tests, subjects and total linked rows.
-- 19667 lab_ids, 8579 subjects, 42829 rows.
SELECT
    COUNT(DISTINCT LAB_TEST_ID) AS lab_ids,
    COUNT(DISTINCT SUBJECT) AS subjects,
    COUNT(*) AS total
FROM AT_ELECTIVE_SURGERY_MICRO_EKP; -- 19667, 8579, 42829



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


-- =============================================================================
-- REQUESTED COUNTS: Percentage of surgery patients in each hospital (using SURGICAL_AREA)
-- =============================================================================

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


-- =============================================================================
-- REQUESTED COUNTS: TOTAL ELECTIVE SURGERY PATIENTS that were screened before surgery and what they were screened for
-- =============================================================================

SELECT * from AT_ELECTIVE_SURGERY_COHORT
WHERE CULTURE_CODE is in (
    'gbscul',
    'gumcul',
    'mngcul',
    'itucul',
    '9itucs',
    '9mrssc',
    'xincul',
    'neocul',
    'caucul',
    'vrecul',
    'cincul',
    'mrscul',
    '9envcs',
    'crocul',
    'rgns'
)
AND LATEST_COLLECT_DT < SURGERY_START_DT; 