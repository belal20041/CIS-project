import mlflow
class MLflowTracker:
    def __init__(self, experiment_name="Sales_Forecasting"):
        mlflow.set_experiment(experiment_name)
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
