# Architecture

**Subject:** AI-based detection of anomalous and falsified IoT sensor data for smart agriculture
**Dataset:** LoRaWAN greenhouse monitoring datasets, Zenodo record 5793685 (University of Antwerp, IDLab), file
`Greenhouse-1.xlsx`: 27 sensors (AF16 to AF42), tomato greenhouse in Belgium. It has one sheet per sensor with the
columns `Date | Time | Onboard Temperature | Onboard Humidity | RSSI | SNR`, one message about every 5.2 minutes,
and 668,368 messages from 10 April to 23 November 2020. Study window: 17 April 06:38 to 3 July 06:00 (§4.1).

---

## 1. The problem, and how the architecture answers it

The project sheet asks for more than "normal vs abnormal". It asks to **tell apart four cases**:

| Case (project sheet)                      | Label in this project |
|-------------------------------------------|-----------------------|
| normal measurement                        | `normal`              |
| value outlier (aberrant value)            | `outlier`             |
| anomaly caused by the sensor itself       | `fault`               |
| deliberately falsified or injected data   | `falsification`       |

This is done in **two stages**:

```
               ┌──────────────────────────────┐   normal
  each row  ──►│ STAGE 1  DETECTION           ├──────────►  OK
 (sensor, t)   │ "is this measurement suspect?"│
               │ z-score · Isolation Forest ·  │
               │ One-Class SVM · Autoencoder · │
               │ (LSTM-AE) · RF · XGBoost      │
               └──────────────┬───────────────┘
                              │ suspect
                              ▼
               ┌──────────────────────────────┐
               │ STAGE 2  QUALIFICATION       │──►  outlier / fault / falsification
               │ "what kind of anomaly?"       │
               │ Random Forest · XGBoost       │
               └──────────────────────────────┘
```

* Stage 1 can work **without labels** (unsupervised models trained on normal data). That is what matters in
  practice, where new attacks are not known in advance. Supervised RF and XGBoost are also trained in stage 1,
  as an upper bound that shows the gap between "knowing the attacks" and "not knowing them".
* Stage 2 needs labels, which the simulation layer provides.

---

## 2. System context: the IoT architecture being protected

```
 PERCEPTION LAYER              NETWORK LAYER                          APPLICATION LAYER
 ┌───────────────┐  LoRa radio  ┌─────────┐   IP    ┌──────────────┐   ┌─────────────────────────────┐
 │ 27 sensor     │ ───────────► │ LoRaWAN │ ──────► │ network      │──►│ storage → DETECTOR (this    │
 │ nodes AF16-42 │  RSSI / SNR  │ gateway │         │ server       │   │ project) → dashboard/alerts │
 │ temp, humidity│              └─────────┘         └──────────────┘   └─────────────────────────────┘
 └───────────────┘
```

Where each simulated scenario fits on the attack surface (use this table in the literature-review chapter):

| Layer        | Threat                                        | Simulated by        | Category       |
|--------------|-----------------------------------------------|---------------------|----------------|
| Perception   | ageing / calibration loss                     | `drift`             | fault          |
| Perception   | frozen sensor, failing electronics            | `stuck`, `random`   | fault          |
| Perception   | glitch, electromagnetic interference          | `spike`             | outlier        |
| Perception   | physical tampering (sensor moved, heated)     | `offset`            | falsification  |
| Network      | replay of captured messages                   | `replay`            | falsification  |
| Network      | spoofed node / forged payload                 | `injection`         | falsification  |
| Application  | database or API tampering                     | `offset`, `injection` | falsification |

Note for the report: LoRaWAN has frame counters and message integrity codes that block naive replay and forgery
on the radio link. The data-level detector is a second line of defence for when those protections are bypassed:
compromised keys, a compromised node, or tampering after the network server.

---

## 3. Pipeline

```
 data/raw/Greenhouse-1.xlsx
        │
 1 DATA         load · audit · clean · resample to a regular grid · chronological split
        │       → data/processed/{train,val,test}.parquet, reports/data_audit.md
 2 SIMULATION   inject faults and attacks, keep the ground truth
        │       → data/processed/{split}_attacked.parquet, reports/attack_log.csv
 3 FEATURES     causal, relative features per (sensor, time step)
        │       → data/features/*.parquet, models/feature_builder.joblib
 4 MODELS       stage 1 detectors + stage 2 classifiers, thresholds calibrated on validation
        │       → models/stage1/*.joblib, models/stage2/*.joblib
 5 EVALUATION   metrics, per-scenario analysis, end-to-end 4-class result, inference time
        │       → reports/results_*.csv, reports/results_summary.md, reports/figures/*.png
 6 PROTOTYPE    Streamlit dashboard: series, score, alerts with type, custom injection, live replay
                → app/streamlit_app.py
```

Code: one module per layer in `src/agri_anomaly/`, and `pipeline.py` chains them. Scripts, the notebook and the
dashboard call the same functions, so nothing is duplicated.

---

## 4. Layer details

### 4.1 Data layer (`data.py`)

| Step            | What                                                                                      |
|-----------------|-------------------------------------------------------------------------------------------|
| Load            | .xlsx (all sheets), .csv or .parquet. Columns are detected by name or set in `config.yaml`. One sheet per sensor is supported (the sheet name becomes the sensor id). |
| Audit           | messages per sensor, median and p95 interval, gaps over 1 h, duplicates, missing and impossible values → `reports/data_audit.md` |
| Clean           | drop data before 17 April 2020 06:38 (sensors were still indoors), drop duplicates, impossible values → missing, Hampel filter removes real isolated spikes |
| Regular grid    | resample each sensor to the median message interval (5 min expected). Gaps of up to 2 steps are interpolated, longer gaps stay missing. `received` and `interpolated` flags are kept. |
| Split           | chronological on the global timeline, 60 / 20 / 20, same cut dates for all sensors        |

Why clean real spikes **before** simulating: the "normal" reference must really be normal. Otherwise a real
anomaly would be labelled normal and every metric would be biased.

**What the real data looks like (study window)**

| Fact | Value | Consequence |
|------|-------|-------------|
| Sensors brought into the greenhouse | 17 April 2020 06:38 (Zenodo notice) | earlier data dropped (42,383 messages) |
| Sensors removed | 3 July 2020 between 06:00 and 08:00: RSSI rises from −115 to −75 dBm as they are carried to the gateway | later data dropped (183,386 messages). After that only some sensors keep sending, from another place (RSSI −80 to −65 dBm, room-like temperature) |
| Messages kept | 442,586 (27 sensors, about 77 days) | |
| Interval | median 5.2 min, 3.5 % of messages lost one by one | 5-minute grid; single lost messages are interpolated |
| Network outages (no sensor sending) | 8 periods, 212 h in total (e.g. 20–25 May, 2–4 June) | stay missing; 17.5 % of the grid is empty |
| Glitches | 1 impossible temperature (−45 °C), 59 temperature and 87 humidity isolated spikes | removed before the simulation |
| Natural noise | humidity jumps 3–8 %RH between messages when vents or misting run | the spike filter only removes deviations above 3 °C or 10 %RH, so real behaviour is kept |
| Split (60/20/20) | train until 2 June 11:00, validation until 17 June 20:30, test until 3 July | 359k / 120k / 117k grid rows |

### 4.2 Simulation layer (`attacks.py`)

| Scenario   | Category      | What is done to the values                                         | Duration    | Intensity            |
|------------|---------------|---------------------------------------------------------------------|-------------|----------------------|
| spike      | outlier       | + or − k·σ on 1–2 points                                            | 5–10 min    | k = 3–6              |
| random     | fault         | erratic uniform values in mean ± k·σ                                | 30 min–3 h  | k = 2–4              |
| stuck      | fault         | frozen on the last value                                            | 2–12 h      | –                    |
| drift      | fault         | linear bias growing up to k·σ                                       | 6–24 h      | k = 0.5–2            |
| offset     | falsification | constant shift of + or − k·σ (artificial modification)              | 1–6 h       | k = 0.3–1.2          |
| injection  | falsification | fabricated plausible values: the sensor's usual daily profile plus noise, with a smooth start | 3–12 h | – |
| replay     | falsification | real values from the same time of day, d days earlier, re-sent      | 3–12 h      | d = 1–7 days         |

σ = standard deviation of that sensor and variable. Each split gets 12 events per scenario (25 for train), and
5 times more spikes (`count_factor`), because a spike lasts only 1–2 points and the outlier class would otherwise
have too few points to measure. `injection` and `replay` replace the whole message
(temperature and humidity together); the others modify one variable. Events never overlap and are spaced by at
least 3 h. Train, validation and test receive different random events (different seeds). Each event is logged
in `attack_log.csv` (sensor, variables, scenario, category, start, end, number of points, intensity, parameters).

### 4.3 Feature layer (`features.py`)

| Family        | Features (for temperature and humidity)                  | Targets                        |
|---------------|----------------------------------------------------------|--------------------------------|
| Dynamics      | `diff`, `std_s` (30 min), `std_l` (3 h)                  | spike, random, stuck           |
| Own history   | `dev_med` (value − rolling median), `z` (rolling z-score) | spike, offset onset            |
| Cross-sensor  | `xdev` (value − median of all sensors − usual offset of this sensor at this hour), `xdev_mean` | offset, drift, injection, replay |
| Co-movement   | `xdiff` (own change − others' change), `xcorr` (rolling correlation with the others' changes) | injection, replay |
| Persistence   | `flat` (log of the number of identical consecutive values) | stuck                         |
| Network       | `rssi_dev`, `snr_dev`                                    | spoofing (extension)           |
| Context       | `hour_sin`, `hour_cos`, `gap` (time since the last message) | day/night, lost messages    |

Three rules:
1. **Causal**: only past steps of the same sensor, and the other sensors at the same time step. A unit test
   checks that changing the future does not change any past feature.
2. **Relative**: no absolute temperature or humidity level. The test period (summer) is warmer than the training
   period (spring), and absolute levels would cause false alarms.
3. **Fitted on train only**: the "usual offset" of each sensor per hour is learned on the clean training split.

The cross-sensor features are the project's "contextual coherence" extension: a falsified value is often
plausible on its own, but inconsistent with the neighbouring sensors.

### 4.4 Model layer (`models/`)

Common interface: `fit(X, y=None, groups=None)`, `score(X) → higher = more suspicious`,
`predict(X) → 0/1 with the threshold calibrated on validation`.

| Stage | Model             | Type          | Trained on                              | Role                                  |
|-------|-------------------|---------------|-----------------------------------------|---------------------------------------|
| 1     | Rolling z-score   | rule          | nothing (scale only)                    | baseline every model must beat        |
| 1     | Isolation Forest  | unsupervised  | clean train                             | classical ML detector                 |
| 1     | One-Class SVM     | unsupervised  | clean train (subsample)                 | classical ML detector                 |
| 1     | Autoencoder       | unsupervised (DL) | clean train                         | reconstruction-error detector         |
| 1     | LSTM autoencoder  | unsupervised (DL) | clean train, 2 h windows            | optional, temporal patterns           |
| 1     | Random Forest     | supervised    | train with simulated attacks            | upper bound                           |
| 1     | XGBoost           | supervised    | train with simulated attacks            | upper bound                           |
| 2     | Random Forest     | supervised    | anomalous rows of train, by category    | outlier / fault / falsification       |
| 2     | XGBoost           | supervised    | anomalous rows of train, by category    | outlier / fault / falsification       |

### 4.5 Evaluation layer (`evaluate.py`)

| Split            | Used for                                                                 |
|------------------|--------------------------------------------------------------------------|
| train (clean)    | fitting unsupervised models and the feature builder                      |
| train (attacked) | fitting supervised models (stage 1 RF/XGB, stage 2)                      |
| val (attacked)   | choosing each model's threshold (max F1, or a target false-positive rate), choosing the best models |
| test (attacked)  | final numbers only, never used for any choice                            |

Reported on test:
* **Stage 1**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, inference time (µs per sample).
  Accuracy is reported because the sheet asks for it, but it is misleading here: with about 5 % anomalies, a
  model that always says "normal" reaches 95 %. Conclusions should rest on F1, recall and PR-AUC.
* **Per scenario**: point recall, event recall (at least one alert during the event), mean detection delay.
* **Stage 2**: per-class precision, recall and F1, confusion matrix, on the true anomalies.
* **End to end**: stage 1 + stage 2 as a 4-class problem (normal / outlier / fault / falsification): accuracy,
  macro F1, F1 per class, 4×4 confusion matrix. Computed for the best unsupervised detector and for the best
  detector overall.

### 4.6 Prototype (`app/streamlit_app.py`)

Pick a sensor, a variable and a detector, then adjust the threshold. The page shows the received value against
the original one, the true attack periods shaded by category, and the alerts coloured by the type stage 2
predicts. Below that come the anomaly score with its threshold, a precision/recall summary, and the alert table.
Two modes: the simulated test attacks, or "inject my own attack" (choose the scenario, start, duration and
intensity). A "play" button replays the test period as if it were arriving live. The page only loads the saved
models; nothing is retrained.

---

## 5. Cross-cutting rules

* **Reproducibility**: one seed in `config.yaml`, separate derived seeds per split, every parameter in the config,
  and the event log saved with every run.
* **No leakage**: chronological split, scalers, feature offsets and thresholds never see the test split, and
  features are causal.
* **Fair comparison**: the table always says which models used labels (`kind` column).
* **Separation of concerns**: logic in `src/`, thin scripts, and the notebook and dashboard reuse the same functions.
* **Tests**: `pytest` checks the column detection, cleaning, grid, split, every scenario, causality of the
  features and the interface of every model.

---

## 6. Mapping to the project sheet

| Project sheet                                        | Where                                                     |
|------------------------------------------------------|-----------------------------------------------------------|
| Obj. 1 / Step 1: IoT architectures and vulnerabilities | Report chapter, section 2 of this document (attack-surface table) |
| Obj. 2 / Step 2: dataset preparation                 | `data.py`, `reports/data_audit.md`                        |
| Obj. 3: missing data, noise, outliers, time variations | `clean()` (Hampel, ranges), `resample_regular()` (interpolation limit), rolling features |
| Obj. 4 / Step 3: anomaly and falsification scenarios | `attacks.py`, `config.yaml > attacks`, `attack_log.csv`   |
| Obj. 5 / Step 4: IF, OCSVM, RF, XGBoost, AE, LSTM-AE  | `models/`                                                 |
| Obj. 6 / Step 5: accuracy … inference time           | `evaluate.py`, `reports/results_*`                        |
| Obj. 7 / Step 6: prototype                           | `app/streamlit_app.py`                                    |
| Distinguish normal / fault / outlier / falsification | stage 2 + end-to-end evaluation                           |
| Extension: context from other sensors / weather      | cross-sensor features (done); weather: see section 7      |

---

## 7. Limitations and extensions

* The categories are defined by how the data was generated. In reality a frozen value can be a fault or an
  attack. Say so in the report; it is the honest limit of any simulation-based study.
* Rolling windows keep the memory of an event for a while, so some alerts fall just after an event ends. They
  count as false positives at point level. Event-level recall and detection delay give the complementary view.
* Results depend on where the simulated events fall: with another `project.seed`, F1 moves by about ±0.05.
  Report the mean and spread over a few seeds.
* On this dataset the z-score rule (F1 0.58) is as good as the unsupervised ML detectors (0.55). The sensors
  agree closely with each other, so cross-sensor deviations are already very informative. Only the supervised
  models clearly do better (0.80).
* Unsupervised detectors are weakest on `injection` (plausible values). This is expected, and it is the
  interesting result to discuss: cross-sensor features are what catch it.
* **Weather extension**: add the outdoor temperature and solar radiation for Hoogstraten (Belgium, where the
  greenhouse is) from a historical weather API (for example Open-Meteo), resampled to the same grid, as extra
  context features.
* **Relocation / spoofing evidence**: the real removal of the sensors on 3 July shows up as a 40 dB RSSI jump,
  which shows that radio metadata can reveal a node that is no longer where it should be.
* **Spoofing extension**: add a scenario that also shifts RSSI and SNR (a fake node transmits from somewhere
  else). The `rssi_dev` and `snr_dev` features are already there.
* **Live stream extension**: replay the test split through MQTT (Mosquitto + paho-mqtt) into the dashboard.
* **Edge deployment**: the z-score rule and the dense autoencoder are the cheapest to score (see the
  `inference_us_per_sample` column), so they are the candidates to run on the gateway.
