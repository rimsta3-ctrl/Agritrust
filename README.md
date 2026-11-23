# AI-based detection of anomalous and falsified IoT sensor data for smart agriculture

Tutored project (projet tutoré). The aim is to detect anomalous or falsified measurements from greenhouse LoRaWAN
sensors **and to say what kind of anomaly each one is**: outlier, sensor fault or falsification.

* Dataset: LoRaWAN greenhouse monitoring datasets, [Zenodo 5793685](https://zenodo.org/records/5793685),
  file `Greenhouse-1.xlsx` (27 sensors, temperature, humidity, RSSI, SNR, April to August 2020).
* Full design: **[ARCHITECTURE.md](ARCHITECTURE.md)**. Read it first; you can reuse it in the report.

```
raw data → 1 DATA → 2 SIMULATION → 3 FEATURES → 4 MODELS (stage 1 detect, stage 2 qualify) → 5 EVALUATION → 6 DASHBOARD
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) without the dataset: synthetic greenhouse data, to check that everything runs (about 3 min)
python scripts/run_all.py --synthetic

# 2) with the real data: put Greenhouse-1.xlsx in data/raw/, then
python scripts/run_all.py

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

### First run on the real file: check the column mapping

The column names of `Greenhouse-1.xlsx` are detected automatically (temp, hum, rssi, snr, time, sensor…).
`01_prepare_data.py` prints the mapping it found:

```
Column mapping detected (first sheet): {'temperature': 'Temperature', 'humidity': ..., 'timestamp': ..., 'sensor_id': ...}
```

If a column is wrong, write its exact name in `config.yaml > data > columns`. If the file has one sheet per
sensor, the sheet name is used as the sensor id. A date and a time in two separate columns are given as
`timestamp: ["Date", "Time"]`. Then open `reports/data_audit.md` and check the median message interval.

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

## What to expect

Results on **synthetic** data are only a check that the code works, not results for the report. In a reference
run (20 synthetic sensors, 60 days):

* supervised XGBoost gets the best F1 (~0.69), and the dense autoencoder is the best unsupervised detector (~0.60);
* spikes, random values, stuck sensors and drift are detected almost always;
* **injection** (fabricated but plausible values) is the hardest case for unsupervised models. That is the
  result to analyse in the report;
* the end-to-end 4-class macro F1 is around 0.5. Separating faults from falsifications is the open challenge
  of the project.

---

## Suggested next steps

1. Run on the real `Greenhouse-1.xlsx`, check the mapping and the audit, and write the data chapter.
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
