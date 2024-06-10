import pandas as pd
import pickle
import os
import argparse
from sklearn.feature_extraction import DictVectorizer

def dump_pickle(obj ,filename):
    with open(filename ,'wb') as f:
        pickle.dump(obj ,f)

def read_dataframe(file_path:str):
    df = pd.read_parquet('data/green_tripdata_2021-01.parquet')
    
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    cols = ['PULocationID' ,'DOLocationID']
    df[cols] = df[cols].astype('str')
    
    return df

def preprocess(df: pd.DataFrame ,dv: DictVectorizer ,fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X ,dv

def run(raw_data_path: str ,dest_path: str ,dataset: str = 'green'):
    
    df_train = read_dataframe(os.path.join(raw_data_path ,f'{dataset}_tripdata_2021-01.parquet'))
    df_val = read_dataframe(os.path.join(raw_data_path ,f'{dataset}_tripdata_2021-02.parquet'))
    df_test = read_dataframe(os.path.join(raw_data_path ,f'{dataset}_tripdata_2021-03.parquet'))
    
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values
    
    dv = DictVectorizer()
    X_train ,dv = preprocess(df=df_train ,dv=dv ,fit_dv=True)
    X_val ,_ = preprocess(df=df_train ,dv=dv ,fit_dv=False)
    X_test,_ = preprocess(df=df_train ,dv=dv ,fit_dv=False)
    
    os.makedirs(dest_path ,exist_ok=True)
    
    dump_pickle(obj=dv ,filename=os.path.join(dest_path ,'dv.pickle'))
    dump_pickle(obj=(X_train ,y_train) ,filename=os.path.join(dest_path ,'train.pickle'))
    dump_pickle(obj=(X_val ,y_val) ,filename=os.path.join(dest_path ,'val.pickle'))
    dump_pickle(obj=(X_test ,y_test) ,filename=os.path.join(dest_path ,'test.pickle'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--raw_data_path',
        help='the location where the raw NYC taxi trip data was saved'
    )
    parser.add_argument(
        "--dest_path",
        help="the location where the resulting files will be saved."
    )
    
    args = parser.parse_args()
    
    run(args.raw_data_path ,args.dest_path)