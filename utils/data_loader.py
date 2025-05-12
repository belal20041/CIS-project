import pandas as pd
class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
    def load_data(self, filename):
        return pd.read_csv(f"{self.data_dir}/{filename}")
