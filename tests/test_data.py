import numpy as np
import pandas as pd

from agri_anomaly import data as D

AUTO = {k: "auto" for k in D.STANDARD_KEYS}


def test_column_detection_with_real_looking_names():
    raw = pd.DataFrame({
        "Device": ["AF 16", "AF 16"],
        "Time": ["2020-04-20 10:00:00", "2020-04-20 10:05:00"],
        "Temperature (°C)": ["21,5", "21,6"],
        "Humidity (%)": [80, 81],
        "RSSI": [-100, -101],
        "SNR": [5.5, 6.0],
    })
    out, used = D.standardize_columns(raw, AUTO)
    assert used["temperature"] == "Temperature (°C)" and used["timestamp"] == "Time"
    assert used["sensor_id"] == "Device"
    assert out["sensor_id"].iloc[0] == "AF16"            # spaces removed
    assert out["temperature"].iloc[0] == 21.5             # decimal comma handled
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


def test_greenhouse1_format_separate_date_and_time():
    """Exact layout of Greenhouse-1.xlsx: one sheet per sensor, Date (datetime) + Time (datetime.time)."""
    import datetime as dt
    raw = pd.DataFrame({
        "Date": pd.to_datetime(["2020-04-17", "2020-04-17", "2020-04-17"]),
        "Time": [dt.time(6, 40, 51), dt.time(6, 46, 2), np.nan],        # rows are not sorted, some times empty
        "Onboard Temperature": [17.5, 17.6, 17.7],
        "Onboard Humidity": [80.1, 80.0, 79.9],
        "RSSI": [-113, -114, -112],
        "SNR": [-6.2, -5.0, -7.1],
    })
    out, used = D.standardize_columns(raw, AUTO, sheet_name="AF16")
    assert used["timestamp"] == ["Date", "Time"]
    assert used["temperature"] == "Onboard Temperature" and used["humidity"] == "Onboard Humidity"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-04-17 06:40:51")
    assert pd.isna(out["timestamp"].iloc[2])
    assert (out["sensor_id"] == "AF16").all()


def test_sheet_name_used_as_sensor_id():
    raw = pd.DataFrame({"timestamp": ["2020-04-20 10:00"], "temp": [20.0], "hum": [70.0]})
    out, _ = D.standardize_columns(raw, AUTO, sheet_name="AF22")
    assert out["sensor_id"].iloc[0] == "AF22"


def test_clean_removes_impossible_values_and_duplicates(cfg):
    ts = pd.date_range("2020-04-18", periods=50, freq="5min")
    df = pd.DataFrame({"timestamp": ts, "sensor_id": "AF16", "temperature": 20.0 + np.sin(np.arange(50) / 5),
                       "humidity": 70.0, "rssi": -100.0, "snr": 5.0})
    df.loc[10, "temperature"] = 85.0                       # impossible
    df = pd.concat([df, df.iloc[[5]]])                     # duplicate
    out, stats = D.clean(df, cfg["data"])
    assert stats["dropped_duplicates"] == 1
    assert stats["temperature_out_of_range_removed"] == 1
    assert out["temperature"].max() < 60


def test_resample_is_regular_and_interpolation_is_limited():
    ts = pd.date_range("2020-04-18", periods=40, freq="5min")
    df = pd.DataFrame({"timestamp": ts, "sensor_id": "AF16", "temperature": np.arange(40.0)})
    df = df.drop(index=[5, 6]).drop(index=range(20, 30))   # short gap (2) and long gap (10)
    out = D.resample_regular(df, "5min", ["temperature"], max_interp_steps=2)
    assert out["timestamp"].diff().dropna().nunique() == 1
    assert out.loc[out["timestamp"] == ts[5], "temperature"].notna().all()     # short gap filled
    assert out.loc[out["timestamp"] == ts[25], "temperature"].isna().all()     # long gap kept empty
    assert not out.loc[out["timestamp"] == ts[5], "received"].iloc[0]


def test_split_is_chronological(grid, cfg):
    splits, _ = D.split_by_time(grid, cfg["data"]["split"])
    assert splits["train"]["timestamp"].max() < splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() < splits["test"]["timestamp"].min()
    assert sum(len(s) for s in splits.values()) == len(grid)
