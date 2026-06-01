import numpy as np
import pandas as pd

def generate_synthetic_esbl(n=2074, seed=42):
    rng = np.random.default_rng(seed)

    # -----------------------
    # NUMERIC FEATURES
    # -----------------------
    age = np.clip(rng.normal(61.2, 14.6, n), 18, 97).round()

    n_sites = rng.choice([1,2,3,4], size=n, p=[0.85, 0.10, 0.04, 0.01])
    n_surgeries = rng.choice(range(1,8), size=n)

    imd = rng.integers(1, 11, size=n).astype(float)
    imd[rng.random(n) < 0.062] = np.nan  # ~6% missing

    def binary(p): return rng.binomial(1, p, n).astype(float)

    df = pd.DataFrame({
        "age_at_admission": age,
        "n_sites": n_sites,
        "n_surgeries": n_surgeries,
        "imd_decile": imd,
        "past_abx": binary(0.31),
        "anemia": binary(0.08),
        "asthma": binary(0.08),
        "cancer": binary(0.22),
        "copd": binary(0.04),
        "hypertension": binary(0.31),
        "ischaemic_heart_disease": binary(0.03),
        "obesity": binary(0.02),
        "renal_failure": binary(0.02),
        "type2_diabetes": binary(0.11),
        "gender": rng.choice([1,2], size=n, p=[0.46, 0.54])
    })

    # -----------------------
    # CATEGORICAL FEATURES
    # -----------------------

    def sample_cat(values, probs):
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        return rng.choice(values, size=n, p=probs)

    df["organism_bug"] = sample_cat(
        ["escherichia coli", "klebsiella pneumoniae", "proteus mirabilis", "klebsiella oxytoca"],
        [0.63, 0.25, 0.076, 0.047]
    )

    df["site"] = sample_cat(
        ["wound","urine","sputum","drain","blood","tips_devices","tissue/biopsy","high_vaginal","low_vaginal"],
        [0.28,0.26,0.13,0.097,0.071,0.058,0.042,0.028,0.020]
    )

    df["tfc_group"] = sample_cat(
        ["abdominal_gi","uro_nephro","cardiothoracic","neuro_ent","general_surgery",
         "gynaecological_oncology","ortho_plastics","vascular","gynaecology","obstetrics"],
        [0.18,0.16,0.11,0.10,0.09,0.09,0.084,0.072,0.053,0.042]
    )

    df["prophylaxis_group"] = sample_cat(
        [
            "cefuroxime | metronidazole",
            "co-amoxiclav (contains penicillin)",
            "cefuroxime",
            "vancomycin",
            "no_prophylaxis",
            "clindamycin",
            "metronidazole",
            "gentamicin | teicoplanin",
            "gentamicin",
            "cefuroxime | gentamicin",
            "cefuroxime | gentamicin | metronidazole",
            "gentamicin | metronidazole"
        ],
        [0.287,0.186,0.145,0.090,0.064,0.061,0.058,0.036,0.031,0.018,0.015,0.016]
    )

    df["ethnicity_desc"] = sample_cat(
        ["white","asian","black"],
        [0.67,0.20,0.13]
    )

    # -----------------------
    # MISSINGNESS
    # -----------------------
    def add_missing(col, rate):
        mask = rng.random(n) < rate
        df.loc[mask, col] = None

    add_missing("site", 0.01)
    add_missing("tfc_group", 0.015)
    add_missing("prophylaxis_group", 0.008)
    add_missing("ethnicity_desc", 0.31)

    # -----------------------
    # TARGET (ESBL)
    # -----------------------
    logit = (
        -1.2
        + 0.02 * df["age_at_admission"]
        + 0.8 * df["past_abx"]
        + 0.6 * df["renal_failure"]
        + 0.5 * df["type2_diabetes"]
        + 0.4 * df["organism_bug"].eq("klebsiella pneumoniae").astype(int)
        + 0.3 * df["site"].isin(["blood","sputum"]).astype(int)
    )

    probs = 1 / (1 + np.exp(-logit))
    probs = probs * 0.4  # match ~34% prevalence
    df["esbl_status"] = rng.binomial(1, probs)

    # -----------------------
    # ADD REALISTIC "MESS"
    # -----------------------

    # duplicates
    dup = df.sample(frac=0.03, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)

    return df



if __name__ == "__main__":
    df = generate_synthetic_esbl()
    df.to_csv("synthetic_esbl_data.csv", index=False)