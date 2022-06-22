import pickle
import pandas as pd 
import sys
import numpy as np


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

year = sys.argv[1]
month = sys.argv[2]

df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)
print(np.mean(y_pred))

df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
df['pred'] = y_pred

with open('output', 'wb') as output_file:
    df[['ride_id', 'pred']].to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
