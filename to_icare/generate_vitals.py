import pandas as pd
import numpy as np

def make_mock_vitals_df(n_subjects=10, rows_per_subject=5, seed=42, save_to_csv=True, filename="mock_vitals.csv"):
    rng = np.random.default_rng(seed)

    subjects = [f"subj_{i:03d}" for i in range(1, n_subjects + 1)]
    obs_codes = ["hr", "temp"]

    records = []

    for subject in subjects:
        for _ in range(rows_per_subject):
            obs_code = rng.choice(obs_codes)

            if obs_code == "hr":
                value = int(rng.integers(50, 121))  # heart rate
                unit = "beats/minute"
            else:
                value = round(rng.uniform(36.0, 39.5), 1)  # temperature
                unit = "degrees celsius"

            performed_dt = pd.Timestamp("2024-01-01") + pd.to_timedelta(
                int(rng.integers(0, 365 * 24 * 60)), unit="m"
            )

            records.append({
                "subject": subject,
                "observation_code": obs_code,
                "observation_performed_dt": performed_dt,
                "observation_result_clean": value,
                "observation_unit": unit
            })

    mock_vitals_df = pd.DataFrame(records).sort_values(
        ["subject", "observation_performed_dt"]
    ).reset_index(drop=True)

    if save_to_csv:
        mock_vitals_df.to_csv(filename, index=False)

    return mock_vitals_df

if __name__ == "__main__":
    make_mock_vitals_df()