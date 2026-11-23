# AI-based detection of anomalous and falsified IoT sensor data for smart agriculture

Tutored project (projet tutoré). The aim is to detect anomalous or falsified measurements from greenhouse LoRaWAN
sensors **and to say what kind of anomaly each one is**: outlier, sensor fault or falsification.

* Dataset: LoRaWAN greenhouse monitoring datasets, [Zenodo 5793685](https://zenodo.org/records/5793685),
  file `Greenhouse-1.xlsx` (32.1 MB, MD5 `4bf157516b3b576abe30550c65ce75d9`): one sheet per sensor (AF16 to AF42),
  columns `Date | Time | Onboard Temperature | Onboard Humidity | RSSI | SNR`, 668,368 messages.
  Study window: **17 April 2020 06:38 to 3 July 2020 06:00**, the period when all 27 sensors were in the
  greenhouse (see `reports/data_audit.md`).
* Full design: **[ARCHITECTURE.md](ARCHITECTURE.md)**. Read it first; you can reuse it in the report.

```
raw data → 1 DATA → 2 SIMULATION → 3 FEATURES → 4 MODELS (stage 1 detect, stage 2 qualify) → 5 EVALUATION → 6 DASHBOARD
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# real data: data/raw/Greenhouse-1.xlsx (about 4 min; the first read of the .xlsx takes about 1 min)
python scripts/run_all.py

# without the dataset: synthetic greenhouse data, only to check the code
python scripts/run_all.py --synthetic

# dashboard
streamlit run app/streamlit_app.py

# tests
pytest -q
```

You can also run the stages one by one, which is useful when you change only one part:

| Script                          | Stage | Main outputs                                                        |
|---------------------------------|-------|---------------------------------------------------------------------|
| `scripts/01_prepare_data.py`    | data  | `data/processed/{train,val,test}.parquet`, `reports/data_audit.md`  |
| `scripts/02_simulate_attacks.py`| simulation | `data/processed/*_attacked.parquet`, `reports/attack_log.csv`  |
| `scripts/03_build_features.py`  | features | `data/features/*.parquet`, `models/feature_builder.joblib`       |
| `scripts/04_train_models.py`    | training | `models/stage1/*.joblib`, `models/stage2/*.joblib`, `reports/training_summary.csv` |
| `scripts/05_evaluate.py`        | evaluation | `reports/results_summary.md`, `reports/results_*.csv`, `reports/figures/*.png` |

`reports/results_summary.md` holds the tables ready to paste into the report.

### The dataset file

`data/` is excluded from Git, so the dataset is not pushed to GitHub. If `data/raw/Greenhouse-1.xlsx` is
missing (for example after a fresh `git clone`), download it from Zenodo into `data/raw/`. The column mapping
for this file is already set in `config.yaml > data > columns`. For another file, set the columns back to `auto`
or write the exact names.

### On Kaggle

1. Push this folder to GitHub, or upload it as a Kaggle dataset.
2. In a notebook: `!git clone https://github.com/<you>/agri-iot-anomaly.git` then `%cd agri-iot-anomaly`.
3. Add the Zenodo dataset as a Kaggle input, and set `paths.raw_file` in `config.yaml` to
   `/kaggle/input/<name>/Greenhouse-1.xlsx`.
4. Follow `notebooks/01_quickstart.ipynb`. Kaggle already has most packages; `pip install -r requirements.txt`
   adds the rest. The dashboard has to be run on your own computer, because Kaggle does not serve Streamlit.

---

## Project structure

```
agri-iot-anomaly/
├── ARCHITECTURE.md          full design (layers, scenarios, features, models, evaluation protocol)
├── config.yaml              every path and parameter (seed, scenarios, windows, models, threshold)
├── requirements.txt
├── src/agri_anomaly/
│   ├── data.py              load, audit, clean, resample, split
│   ├── synthetic.py         synthetic greenhouse data (for testing without the dataset)
│   ├── attacks.py           7 scenarios → 3 categories, event log = ground truth
│   ├── features.py          causal, relative features (own history + cross-sensor coherence)
│   ├── models/
│   │   ├── base.py          common interface fit / score / predict
│   │   ├── classical.py     z-score baseline, Isolation Forest, One-Class SVM, RF, XGBoost, stage-2 classifier
│   │   └── deep.py          dense autoencoder, LSTM autoencoder (PyTorch)
│   ├── evaluate.py          thresholds, metrics, per scenario, stage 2, end to end
│   ├── plots.py             figures for the report
│   └── pipeline.py          the 5 stages chained (used by scripts, notebook, dashboard)
├── scripts/                 01 … 05 + run_all.py
├── app/streamlit_app.py     dashboard
├── notebooks/01_quickstart.ipynb
├── tests/                   pytest
├── data/{raw,processed,features}/
├── models/                  saved models
└── reports/                 audit, results tables, figures
```

---

## Results on the real data (reference run, seed 42, test split 17 June to 3 July 2020)

Full tables: `reports/results_summary.md`. Figures: `reports/figures/`.

| Stage-1 detector  | Type         | Precision | Recall | F1   | PR-AUC | False-positive rate |
|-------------------|--------------|-----------|--------|------|--------|---------------------|
| Random Forest     | supervised   | 0.84      | 0.77   | 0.81 | 0.85   | 0.9 %               |
| XGBoost           | supervised   | 0.81      | 0.79   | 0.80 | 0.85   | 1.2 %               |
| z-score baseline  | rule         | 0.66      | 0.52   | 0.58 | 0.62   | 1.7 %               |
| Autoencoder       | unsupervised | 0.55      | 0.55   | 0.55 | 0.46   | 2.7 %               |
| One-Class SVM     | unsupervised | 0.51      | 0.59   | 0.55 | 0.51   | 3.5 %               |
| Isolation Forest  | unsupervised | 0.26      | 0.59   | 0.36 | 0.32   | 10.5 %              |

* Almost every event is caught by at least one alert (event recall 0.83 to 1.0 for all ML models). The point-level
  scores are lower, because the first and last steps of an event are hard to separate from normal data.
* Stage 2 (type of anomaly) reaches about 81 % accuracy on the true anomalies (macro F1 0.81 to 0.83).
* End to end, on 4 classes (normal / outlier / fault / falsification): macro F1 0.77 with XGBoost in stage 1, and
  0.67 with the best unsupervised detector. Telling faults from falsifications is the hardest part.
* Points to discuss in the report: the simple z-score rule is as good as the unsupervised ML models here, and
  the scores move by about ±0.05 when the attacks are placed differently (`project.seed`). Report a mean over a
  few seeds.

## Suggested next steps

1. Read `reports/data_audit.md` (study window, network outages, cleaning) and write the data chapter.
2. Validate the scenario-to-category table (ARCHITECTURE.md §4.2) with your supervisor.
3. Tune: feature windows, scenario intensities, model parameters, threshold strategy (`f1` or `fpr`).
4. Turn on the LSTM autoencoder (`config.yaml > models > stage1 > lstm_autoencoder > enabled: true`).
5. Extensions: weather context, a spoofing scenario using RSSI, an MQTT live stream (ARCHITECTURE.md §7).

### Splitting the work between two people

| Person A                                         | Person B                                        |
|--------------------------------------------------|-------------------------------------------------|
| literature review: IoT architecture and attacks  | literature review: anomaly-detection methods    |
| data layer and audit, attack scenarios           | features, models, tuning                        |
| dashboard                                        | evaluation and figures                          |
| shared: report, presentation, demo               |                                                 |

Work on separate Git branches. Both of you use the same `config.yaml`, so the results stay comparable.
