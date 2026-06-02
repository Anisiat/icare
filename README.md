# 🧫 ESBL Epidemiology & Risk Factors Pipeline

This project investigates **epidemiology**, **risk factors**, and **predictive modeling** for **ESBL infections** using linked hospital datasets (NHS–Snowflake) and MIMIC-IV.

The workflow is designed as a **reproducible** and **modular** pipeline:

1. Extract raw data (Snowflake / MIMIC-IV)
2. Clean and harmonize source tables (demographics, microbiology, problems, surgeries)
3. Merge into a unified patient-level analysis dataset
4. Generate descriptive statistics (e.g., Table 1)
5. Feature engineering for machine learning
6. (Planned) Model training and evaluation

---

## 🗂 Project Structure

```

esbl_project/
├── data/
│   ├── raw/         # Original data extracts (never modified)
│   ├── interim/     # Cleaned / intermediate outputs
│   └── processed/   # Final analysis & ML datasets
│
├── src/
│   ├── utils/       # Reusable helper functions
│   ├── steps/       # Pipeline stages (cleaning, merging, feature creation)
│   └── analysis/    # Statistical summaries & EDA (e.g., Table 1 scripts)
│
├── scripts/
│   └── run_pipeline.py    # Master script to execute the workflow
│
├── notebooks/      # Exploration and validation
├── configs/
│   └── config.yaml # Central configuration: paths + settings
└── README.md       # 📌 You are here

```

---

## ⚙️ Configuration

All file paths and settings are stored centrally in:

```

configs/config.yaml

````

Example:

```yaml
paths:
  demographics: data/raw/demographics.csv
  microbiology: data/raw/microbiology.csv
  problems: data/raw/problems.csv
  surgeries: data/raw/surgeries.csv
  interim_dir: data/interim/
  processed_dir: data/processed/

settings:
  prophylaxis_window_hours: 12
  include_outpatients: false
````

This allows running the pipeline from any working directory without modifying code.

---

## ▶️ Run the Pipeline

### 1️⃣ Install environment

```bash
conda create -n esbl python=3.11 -y
conda activate esbl
pip install -r requirements.txt
```

*(If `requirements.txt` is missing, it will be added soon.)*

### 2️⃣ Place Hospital Data

Copy raw tables into:

```
data/raw/
```

### 3️⃣ Execute the full pipeline

```bash
python scripts/run_pipeline.py
```

Outputs will be written to:

* **data/interim/** → intermediate cleaned tables
* **data/processed/** → final unified and ML-ready datasets

---

## 🔁 Running Individual Steps

Each stage is modular and can be run independently for debugging:

```bash
python src/steps/clean_demographics.py
```

Each step:

* Loads inputs defined in `config.yaml`
* Writes outputs to `data/interim/`
* Can be imported from notebooks (`from src.steps.clean_demographics import run`)

---

## 📊 Outputs

✔ Merged cohort table
✔ Cleaned demographics / microbiology / problems / surgeries
✔ ESBL phenotype inference
✔ Prophylaxis inference with decision logic
✔ Summary descriptive statistics (Table 1)
✔ Feature sets for predictive modeling

---

## 🧪 Testing (Future Work)

Planned enhancements:

* Unit tests for utils + data validation (pytest)
* Automated checks: key uniqueness, missing data, type consistency
* Continuous-integration workflow

---

## 📌 Citation (Optional Future Section)

A citation section will be added once study results are published.

---

```

---

If you'd like, I can also:

✨ Add a workflow **diagram** for the README  
✨ Auto-generate a **requirements.txt** from your imports  
✨ Add project **badges** (Python version, license, CI)  
✨ Add a **Data Dictionary** section  
✨ Add a **Changelog** once versions evolve  

Just tell me what you want next!
```




GET TREE FUNCTION 

python3 - << 'EOF'
import os

def tree(dir_path, prefix=""):
    contents = sorted(os.listdir(dir_path))
    for index, name in enumerate(contents):
        full_path = os.path.join(dir_path, name)
        connector = "└── " if index == len(contents) - 1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(full_path):
            extension = "    " if index == len(contents) - 1 else "│   "
            tree(full_path, prefix + extension)

tree('.', '')
EOF