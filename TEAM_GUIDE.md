# Team guide: how to build on this repo (G154 / MP213)

Read this before you write any code. It exists so that we do not overwrite each
other's work. The data part is done and pushed; this repo is the **base** for
everything else.

## 1. The golden rules

1. **Never `git push --force`** (or `-f`). It deletes your teammates' commits.
2. **Always `git pull --rebase origin main` before you push.**
3. **Never `git init` a new repo** or push a different history. Always work in a clone of this one.
4. **Commit from your own GitHub account** (see section 2). Commits are counted per person for the marks.
5. **Never open `data/raw/train.csv` in Excel and save it.** Its hash must stay
   `1cf325bb34ff663d4883d2e6b40dba18ff5b7b5ce0e9b92f7087326142c64ee4` (SHA-256).
6. **Work on your own branch**, and merge by Pull Request (section 4).

## 2. First-time setup

```powershell
git clone https://github.com/Ishitabora/mlops.git
cd mlops
git config user.name  "<your GitHub name>"
git config user.email "<your GitHub email>"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Getting the data (there is no DVC remote)
`dvc pull` will not work: the CSVs are not stored on GitHub, only their `.dvc`
fingerprints. Copy the **original** files from your own zip into `data/raw/`:

```
data/raw/train.csv
data/raw/later.csv
data/raw/holdout_features.csv     (already in git)
```

Then check you have the right files:

```powershell
dvc status                       # must say: Data and pipelines are up to date
python -c "import hashlib;print(hashlib.sha256(open('data/raw/train.csv','rb').read()).hexdigest())"
```

If `dvc status` reports a change, your file differs from ours. Re-extract it from the zip.

## 3. What is already built (do not redo it)

| Thing | Where | Notes |
|---|---|---|
| The 7 data checks | `src/bakery/checks.py` | `run_all(train, later, holdout, params)` returns the findings |
| Cleaning | `src/bakery/clean.py` | `clean_train(train, issues, params)` returns the cleaned table |
| Runner | `src/bakery/run_data_checks.py` | writes `data/processed/clean_train.csv` |
| Settings | `params.yaml` | paths, column lists, thresholds. Add your own keys here |
| Data tests | `tests/test_data_checks.py` | 6 tests, no CSVs needed |
| `findings.yaml` | repo root | `data_issues` is **filled in**; `tracking` and `monitoring` are yours |

### Use the cleaned data like this (do not write your own cleaning)

```python
import pandas as pd, yaml
from bakery.checks import run_all
from bakery.clean import clean_train

params = yaml.safe_load(open("params.yaml"))
d = params["data"]
train, later, holdout = (pd.read_csv(d[k]) for k in ("train", "later", "holdout"))
issues = run_all(train, later, holdout, params)
clean = clean_train(train, issues, params)      # 1000 rows, no blanks, no leaking column
```

Facts you need:

- **Features** are the 8 columns in `params.yaml`: 6 numeric + `day_type`, `outlet`. Target is `loaves_sold`.
- **Never use `loaves_in_closing_register`.** It leaks the answer and is missing from the holdout file. `clean_train` already drops it.
- `later.csv` is clean already. Do not run `clean_train` on it.
- Run from the repo root with `PYTHONPATH=src` (or add `src` to your path in tests).

## 4. Who does what

| Member | Owns | Files (create these) | Marks |
|---|---|---|---|
| Data (done) | data checks, DVC, cleaning | `src/bakery/checks.py`, `clean.py`, `data/raw/*.dvc` | C, D |
| Model + monitoring | training, MLflow, registry, PSI, v2, holdout | `src/bakery/train.py`, `src/bakery/monitor.py`, `mlflow.db`, `models/model.joblib`, `predictions/holdout_predictions.csv` | E, K, L |
| Deployment | API, Docker, CI, k8s, logging | `app/main.py`, `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, `k8s/deployment.yaml`, `k8s/service.yaml` | F, G, H, I, J |

Why this stops conflicts: **each person writes new files in their own area.** Edit a
shared file only for the small changes listed in section 5.

If there is a fourth member, split Deployment: one takes `app/` + tests + Docker, the other CI + k8s + logging.

### Model + monitoring: things to get right
- `mlflow.set_tracking_uri("sqlite:///mlflow.db")`; log param `data_sha256` (the hash above) in **every** run.
- Log a metric with `mae` in its name; run 3+ settings; register the model; v1 = version 1, v2 = version 2; alias `champion`.
- `later.csv` already has `shelf_hours` much higher than `train.csv` (mean about 16 vs 10). Check PSI carefully.
- Promote v2 only if its MAE on the last 30% of `later.csv` is **at least 12% lower** than v1's.
- `mlflow.db` is a binary file and **cannot be merged**. Only the model owner commits it; everyone else leaves it alone.

### Deployment: things to get right
- Request fields exactly: `price_rs, temp_c, footfall_index, competitor_shops, shelf_hours, orders_previous_day, day_type, outlet`.
- Put the rules in the pydantic schema: `shelf_hours` `le=25`, `temp_c` `ge=2.5` (bad input gives 422).
- Image `mp213-api:1.0`, Deployment `mp213-api`, Service `mp213-svc`, 3 replicas, container port 8000, service port 80.
- Tests must **not need the CSVs**. Build small data inside the test. At least one test uses `TestClient`.
- `ci.yml` must start with `on:` and run `pytest` on push. Get a green tick.

## 5. Shared files: how to edit without clashing

| File | Rule |
|---|---|
| `requirements.txt` | **Add** your lines at the bottom, one per line, pinned (`mlflow==x.y.z`). Do not reformat or reorder. Resolve a conflict by keeping both sides' lines. |
| `params.yaml` | Add your own top-level section (`model:`, `api:` ...). Do not edit the `data:` or `checks:` sections. |
| `findings.yaml` | Fill in only your section: `tracking` + `monitoring` (model owner). Do not touch `data_issues`. |
| `README.md` | Add a short section for your part at the bottom. |
| `.gitignore` | Append only. |

Keep **all** libraries pinned in `requirements.txt`. If you `pip install` something new, add it.

## 6. Daily workflow (copy this)

```powershell
git checkout main
git pull --rebase origin main            # get everyone's latest work
git checkout -b feat/<your-part>         # e.g. feat/model, feat/deploy

# ... work, then:
git add <the files you changed>          # avoid "git add ." so you do not stage stray files
git commit -m "feat(model): train three models and log to MLflow"
git pull --rebase origin main            # again, in case someone pushed meanwhile
git push -u origin feat/<your-part>
```

Then open a **Pull Request** on GitHub into `main`, and ask one teammate to review and merge it. After it is merged:

```powershell
git checkout main
git pull --rebase origin main
```

Small commits, pushed often, mean small conflicts.

### If `git push` is rejected
It means someone pushed first. **Do not force.** Run:

```powershell
git pull --rebase origin main
# fix any conflict markers in the files, keep both sides' content, then:
git add <fixed files>
git rebase --continue
git push
```

### If you see "unrelated histories" or "no common ancestor"
Stop. You created a new repo instead of cloning this one. Delete your folder and clone again (section 2).

## 7. Before you submit (from the handbook, section 13)

- [ ] Repo is public and opens in a private browser window.
- [ ] README has the group id, problem id and every enrolment number.
- [ ] `findings.yaml` has no `null` left except where a problem is absent. **Move it to `reports/findings.yaml`** (the layout the marking tool reads), once all sections are filled.
- [ ] `mlflow.db` is committed and the `champion` alias is set.
- [ ] The Actions tab shows a green tick.
- [ ] `predictions/holdout_predictions.csv` has every `record_id`.
- [ ] Every member has commits under their own name.
- [ ] **Every member fills the submission form** with the same repository link.
- [ ] Every member can explain every file. The viva asks you to change your own code live.

**Deadline: Sunday 29 Nov 2026, 11:59 PM** (grace period to Fri 4 Dec costs 1 mark).
