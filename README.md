# MP213 - Bakery Bread Sales (Kochi)

**Group:** G154  **Problem:** MP213  **Course:** SCSE3040 MLOps  **Faculty:** Ravi Pratap  **Batch:** 65

| Enrolment no. | Name |
|---|---|
| S24CSEU2463 | Ishitabora |
| S24CSEU2465 | JatinDhaked |
| S24CSEU2426 | ArpitSehrawat |
| S24CSEU2430 | Yashupadhyay |

Predicts `loaves_sold` per outlet per day for the Kochi Bakers' Guild (regression, MAE).

## Data pipeline

```
data/raw/*.csv  --DVC-->  7 data checks  -->  cleaning  -->  data/processed/clean_train.csv
```

- `src/bakery/checks.py` - the seven data checks (leak, unit mix-up, sentinel, duplicates, stuck sensor, target typos, spelling).
- `src/bakery/clean.py` - fixes every problem the checks find.
- `src/bakery/run_data_checks.py` - runs checks, prints findings, writes the cleaned file.
- `params.yaml` - paths and thresholds.

## Run

```
pip install -r requirements.txt
dvc pull                                   # fetch data/raw/train.csv and later.csv
PYTHONPATH=src python -m bakery.run_data_checks
PYTHONPATH=src python -m pytest tests -q
```

Never open or re-save `data/raw/train.csv` in Excel: its hash would change.

## Part 2: Model Training, Tracking, and Drift Monitoring

- `src/bakery/train.py` - Cleans training data, tracks candidate models in MLflow (`sqlite:///mlflow.db`) with `data_sha256`, registers the best candidate as version 1, and sets the `@champion` alias.
- `src/bakery/monitor.py` - Computes 10-bin PSI drift per numeric column (`shelf_hours` drifted most), evaluates subgroup MAE across outlets, retrains v2 on clean train + first 70% of `later.csv`, evaluates the >= 12% promotion threshold on the last 30% check set (promoted to `@champion`), and outputs `predictions/holdout_predictions.csv`.
- `tests/test_model.py` - Self-contained unit tests for pipeline fitting/inference, unseen category handling, PSI calculation, promotion decision rule, and holdout prediction formatting without needing raw CSVs.

