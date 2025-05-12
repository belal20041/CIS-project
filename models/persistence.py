import joblib
class ModelPersistence:
    def save_model(self, model, path):
        joblib.dump(model, path)
    def load_model(self, path):
        return joblib.load(path)
