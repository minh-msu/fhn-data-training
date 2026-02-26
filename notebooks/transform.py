import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

def transform(df, label=None):
    df['pickup_datetime'] = pd.to_datetime(
        df['pickup_datetime'], 
        utc=True
    )
    df['local_datetime'] = df['pickup_datetime'].dt.tz_convert(
        ZoneInfo("America/New_York")
    )
    df.sort_values(
        by='local_datetime', 
        ascending=True, 
        inplace=True
    )
    timestamp = df['local_datetime']
    df['dayofweek'] = timestamp.dt.dayofweek
    df['day'] = timestamp.dt.day
    df['month'] = timestamp.dt.month
    df['year'] = timestamp.dt.year 
    df["hour"] = timestamp.dt.hour
    df["minute"] = timestamp.dt.minute
    df["second"] = timestamp.dt.second

    d_lat = df['dropoff_latitude'] - df['pickup_latitude']
    d_lon = df['dropoff_longitude'] - df['pickup_longitude']
    df['distance'] = np.sqrt(d_lat**2 + d_lon**2)
    df['bearing'] = (np.degrees(np.arctan2(d_lat, d_lon)) + 360) % 360
    df = df.dropna()

    drop_cols = ['key', 'pickup_datetime', 'local_datetime']
    X = df.drop(drop_cols, axis=1)
    X = X.astype(float)
    if label:
        X.drop(label, axis=1, inplace=True)
        y = df[label]
        return X, y 
    return X