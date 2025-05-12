from xgboost import XGBRegressor
class ModelTrainer:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model
