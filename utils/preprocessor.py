import pandas as pd
class Preprocessor:
    def clean_data(self, df):
        return df.dropna()
    def engineer_features(self, df):
        return df
