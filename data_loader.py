import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, file_path='horse.csv'):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = []
        self.numerical_features = []
        self.categorical_features = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputers = {}

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/HorseColic/horse.csv"
            self.data = pd.read_csv(url)
        return self.data

    def preprocess_data(self):
        if 'outcome' in self.data.columns:
            self.y = self.data['outcome'].copy()
            self.X = self.data.drop('outcome', axis=1)
        else:
            raise ValueError("Target column 'outcome' not found in dataset")

        if 'hospital_number' in self.X.columns:
            self.X = self.X.drop('hospital_number', axis=1)
            print("Removed 'hospital_number' from features")

        self.numerical_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()

        if self.numerical_features:
            self.imputers['numerical'] = SimpleImputer(strategy='mean')
            self.X[self.numerical_features] = self.imputers['numerical'].fit_transform(
                self.X[self.numerical_features]
            )

        if self.categorical_features:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            self.X[self.categorical_features] = self.imputers['categorical'].fit_transform(
                self.X[self.categorical_features]
            )

        for col in self.categorical_features:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            self.label_encoders[col] = le

        if self.y.dtype == 'object':
            le_target = LabelEncoder()
            self.y = le_target.fit_transform(self.y)
            self.label_encoders['target'] = le_target

        self.feature_names = self.X.columns.tolist()

        return self.X, self.y

    def get_feature_stats(self):
        stats = {}
        for col in self.X.columns:
            if col in self.numerical_features:
                if col == 'total_protein':
                    stats[col] = {'type': 'numerical', 'default': 7.0}
                else:
                    stats[col] = {'type': 'numerical', 'default': self.X[col].mean()}
            else:
                stats[col] = {'type': 'categorical', 'default': self.X[col].median()}
        return stats
