-- ============================================================================
-- MICROBIOLOGY EKP TABLE DERIVATION
-- Source: Transcribed and annotated from screenshot.
-- Purpose:
--   Build a microbiology dataset containing:
--     * EKP organisms (E. coli, Klebsiella spp., Proteus mirabilis)
--     * Antibiotic susceptibility test (AST) results
--     * ESBL marker information
--     * Culture type information
--
-- Notes:
--   - ESBL status is inferred from susceptibility records and lookup tables.
--   - Antibiotic/result strings are split into separate fields.
--   - Antibiotics excluded from the final antibiogram are removed.
--   - AST results are pivoted so each antibiotic becomes a column.
-- ============================================================================


-- Quick QA counts
SELECT
    COUNT(DISTINCT LAB_TEST_ID) AS lab_ids,
    COUNT(DISTINCT SUBJECT) AS subjects,
    COUNT(*) AS total
FROM at_microbiology_ekp;


-- Inspect sample records
SELECT *
FROM at_microbiology_ekp
LIMIT 400;


CREATE OR REPLACE TABLE AT_MICROBIOLOGY_EKP AS (

WITH microbiology_ekp_cte AS (

    SELECT *
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON

    WHERE LOWER(organism_bug) IN (
        'escherichia coli',
        'klebsiella pneumoniae',
        'klebsiella oxytoca',
        'proteus mirabilis'
    )

    AND (

        ------------------------------------------------------------------------
        -- Condition 1:
        -- Organism has a susceptibility result contained in the
        -- cephalosporin-resistant lookup table.
        ------------------------------------------------------------------------
        LOWER(sensitivity) IN (
            SELECT DISTINCT sensitivity
            FROM ICHT_SANDBOX_PROD.SMTPATH_25046.LOOKUP_CEPHALOSPORIN_R_AT
        )

        OR

        ------------------------------------------------------------------------
        -- Condition 2:
        -- Organism has a susceptibility result contained in the
        -- cephalosporin-sensitive lookup table.
        ------------------------------------------------------------------------
        LOWER(sensitivity) IN (
            SELECT DISTINCT sensitivity
            FROM ICHT_SANDBOX_PROD.SMTPATH_25046.LOOKUP_CEPHALOSPORIN_S_AT
        )

        OR

        ------------------------------------------------------------------------
        -- Condition 3:
        -- Explicit ESBL marker present.
        -- Exclude "not determined" records.
        ------------------------------------------------------------------------
        (
            LOWER(sensitivity) LIKE '%esbl markers%'
            AND LOWER(sensitivity) NOT LIKE '%not determined%'
        )

        OR

        ------------------------------------------------------------------------
        -- Condition 4:
        -- Include cefpodoxime susceptibility records.
        ------------------------------------------------------------------------
        LOWER(sensitivity) LIKE '%cefpodoxime%'
    )
),


-- ============================================================================
-- Extract all antibiotic susceptibility results from qualifying lab tests.
-- Split combined sensitivity string into:
--   antibiotic_name
--   sensitivity_result
-- Example:
--   "Co-amoxiclav : R"
-- becomes:
--   antibiotic_name = "Co-amoxiclav"
--   sensitivity_result = "R"
-- ============================================================================
split_ast_cte AS (

    SELECT
        SUBJECT,
        LAB_TEST_ID,
        LATEST_COLLECT_DT,
        LATEST_RECEIVED_DT,
        LATEST_RESULT_DT,
        SITE,
        ORGANISM_BUG,

        TRIM(SPLIT_PART(sensitivity, ' : ', 1)) AS antibiotic_name,
        TRIM(SPLIT_PART(sensitivity, ' : ', 2)) AS sensitivity_result

    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON

    WHERE LAB_TEST_ID IN (
        SELECT DISTINCT LAB_TEST_ID
        FROM microbiology_ekp_cte
    )

      AND ORDER_NAME = 'susceptibility'

      AND LOWER(organism_bug) IN (
          'escherichia coli',
          'klebsiella pneumoniae',
          'klebsiella oxytoca',
          'proteus mirabilis'
      )
),


-- Remove antibiotics excluded from the final antibiogram
split_ast_clean_cte AS (

    SELECT *
    FROM split_ast_cte

    WHERE antibiotic_name NOT IN (
        SELECT ANTIBIOTIC
        FROM AT_ABX_EXCLUDE
    )
),


-- ============================================================================
-- Pivot AST results:
-- Rows:
--     LAB_TEST_ID + organism
-- Columns:
--     Antibiotics
-- Values:
--     S / I / R etc.
-- ============================================================================
pivot_ekp_ast_cte AS (

    SELECT *
    FROM split_ast_clean_cte

    PIVOT (

        -- Ideally one result per antibiotic per test.
        MAX(sensitivity_result)

        FOR antibiotic_name IN (
            SELECT antibiotic_name
            FROM split_ast_clean_cte
        )
    )
),


-- ============================================================================
-- Capture culture type information for the selected laboratory tests.
-- ============================================================================
order_name_table AS (

    SELECT
        LAB_TEST_ID,
        ORDER_NAME AS CULTURE_TYPE,
        ORDER_CODE AS CULTURE_CODE

    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON

    WHERE LAB_TEST_ID IN (
        SELECT DISTINCT LAB_TEST_ID
        FROM pivot_ekp_ast_cte
    )

      AND ORDER_NAME != 'susceptibility'

      AND LOWER(organism_bug) IN (
          'escherichia coli',
          'klebsiella pneumoniae',
          'klebsiella oxytoca',
          'proteus mirabilis'
      )

      -- Original note from screenshot:
      -- "what do I do about samples that also have other bugs?"
)


-- ============================================================================
-- Final output table
-- ============================================================================
SELECT
    m.*,
    o.culture_type,
    o.culture_code

FROM pivot_ekp_ast_cte m

LEFT JOIN order_name_table o
    ON o.LAB_TEST_ID = m.LAB_TEST_ID

ORDER BY LAB_TEST_ID

);
