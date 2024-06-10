# Mlops-hw2
```
    import dagshub
    dagshub.init(repo_owner='akesherwani900', repo_name='Mlops-hw2', mlflow=True)

    import mlflow
    with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
```
