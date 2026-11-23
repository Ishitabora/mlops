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
