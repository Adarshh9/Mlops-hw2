import os
import argparse
import pickle
import warnings
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error ,r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def load_pickle(filepath: str):
    with open(filepath ,'rb') as f:
        file = pickle.load(f)
    return file

def run(data_path: str ,max_depth: int ,n_estimators: int):
    
    X_train ,y_train = load_pickle(filepath=os.path.join(data_path ,'train.pickle'))
    X_val ,y_val = load_pickle(filepath=os.path.join(data_path ,'val.pickle'))
        
    dagshub.init(repo_owner='akesherwani900', repo_name='Mlops-hw2', mlflow=True)
    
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        model.fit(X_train ,y_train)
        
        y_val_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val ,y_val_pred)
        r2 = r2_score(y_val ,y_val_pred)
        
        print(f'Random Forest model (max_depth={max_depth} ,n_estimators={n_estimators}):')
        print(f'RMSE >> {rmse}')
        print(f'R2 Score >> {r2}')
        mlflow.log_param('n_estimators',n_estimators)
        mlflow.log_param('max_depth' ,max_depth)
        mlflow.log_metric('rmse' ,rmse)
        mlflow.log_metric('r2_score' ,r2)
        
        mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestHW2Model")
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_path',
        default='./output',
        help="the location where the processed NYC taxi trip data was saved."
    )
    
    parser.add_argument(
        '--max_depth',
        default=10,
        help="hyperparameter for Random Forest Regressor model."
    )
    
    parser.add_argument(
        '--n_estimators',
        default=100,
        help="hyperparameter for Random Forest Regressor model."
    )
    
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")
    
    run(args.data_path ,int(args.max_depth) ,int(args.n_estimators))