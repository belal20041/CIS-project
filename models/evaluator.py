from sklearn.metrics import mean_squared_error
import numpy as np
class Evaluator:
    def calculate_rmsle(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
