from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


class ModelTrainer:
    def __init__(self, n_estimators=200, max_depth=15, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        self.trained = False

    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_

    def save_model(self, filepath='rf_model.pkl'):
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)


    def load_model(self, filepath='rf_model.pkl'):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True
