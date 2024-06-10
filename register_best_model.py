import os
import argparse
import pickle
import warnings
import numpy as np
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error ,r2_score

import mlflow
import mlflow.sklearn

from hyperopt import STATUS_OK ,Trials ,fmin ,hp  ,tpe
from hyperopt.pyll import scope
import dagshub

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def load_pickle(filepath: str):
    with open(filepath ,'rb') as f:
        file = pickle.load(f)
    return file


def evaluate_and_register_best_model(models_path: str ,outputs_path: str):
    
    outputs = os.listdir(outputs_path)
    
    # file "16-4.pkl" where n_estimators=16 and max_depth=4
    best_rmse = float("inf")
    best_model = None
    for file in outputs:
        with open(f'{outputs_path}/{file}' ,'rb') as f:
            output = pickle.load(f)
        
        if output['loss'] < best_rmse:
            best_rmse = output['loss']
            best_model = output['model_name']
            
    with open(f'{models_path}/{best_model}' ,'rb') as f:
        best_model = pickle.load(f)
            
    mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="RandomForest-HW2-BestModel")
    

def run(data_path: str ,num_trials: int ,models_path: str ,outputs_path: str):
    
    X_train ,y_train = load_pickle(filepath=os.path.join(data_path ,'train.pickle'))
    X_val ,y_val = load_pickle(filepath=os.path.join(data_path ,'val.pickle'))

    dagshub.init(repo_owner='akesherwani900', repo_name='Mlops-hw2', mlflow=True)
        
    def objective(params):
        with mlflow.start_run():
            model = RandomForestRegressor(
                n_estimators=int(params['n_estimators']), 
                max_depth=int(params['max_depth']), 
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                random_state=params['random_state']
            )
            model.fit(X_train ,y_train)
            
            y_val_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val ,y_val_pred)
            r2 = r2_score(y_val ,y_val_pred)
            
            print(f'''Random Forest model ( max_depth={params["max_depth"]},
                    n_estimators={params["n_estimators"]},
                    min_samples_split={params["min_samples_split"]},
                    min_samples_leaf={params["min_samples_leaf"]},
                    random_state={params["random_state"]}):''')
            print(f'RMSE >> {rmse}')
            print(f'R2 Score >> {r2}\n\n')
            mlflow.log_param('n_estimators',int(params['n_estimators']))
            mlflow.log_param('max_depth' ,int(params['max_depth']))
            mlflow.log_param('min_samples_split' ,int(params['min_samples_split']))
            mlflow.log_param('min_samples_leaf' ,int(params['min_samples_leaf']))
            mlflow.log_param('random_state' ,int(params['random_state']))
            mlflow.log_metric('rmse' ,rmse)
            mlflow.log_metric('r2_score' ,r2)
            
            model_name = f"{params['n_estimators']}-{params['max_depth']}.pkl"
            
            with open(f"artifacts/models/{model_name}" ,"wb") as file:
                pickle.dump(model ,file)
            
            output = {
                'model_name' : model_name,
                'accuracy' : r2,
                'loss' : rmse
            }
            
            output_file_name = f"{params['n_estimators']}-{params['max_depth']}.pkl"
            with open(f'artifacts/outputs/{output_file_name}' ,'wb') as file:
                pickle.dump(output ,file)
            
            return {'accuracy':r2, 'loss': rmse, 'status': STATUS_OK}
        
    search_space = {
        'max_depth' : scope.int(hp.quniform('max_depth' ,1 ,20 ,1)),
        'n_estimators' : scope.int(hp.quniform('n_estimators' ,10 ,50 ,1)),
        'min_samples_split' : scope.int(hp.quniform('min_samples_split' ,2 ,10 ,1)),
        'min_samples_leaf' : scope.int(hp.quniform('min_samples_leaf' ,1 ,4 ,1)),
        'random_state' : 42
    }
    
    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )
    
    evaluate_and_register_best_model(models_path=models_path ,outputs_path=outputs_path)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_path',
        default='./output',
        help="the location where the processed NYC taxi trip data was saved."
    )
    
    parser.add_argument(
        '--num_trials',
        default=10,
        help="number of trials for evaluating model with different hyper params."
    )
    
    parser.add_argument(
        '--models_path',
        default='artifacts/models',
        help="the location where the trained models were saved."
    )
    
    parser.add_argument(
        '--outputs_path',
        default='artifacts/outputs',
        help="the location where outputs of trained models were saved."
    )
    
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")
    
    run(args.data_path ,int(args.num_trials) ,args.models_path ,args.outputs_path)