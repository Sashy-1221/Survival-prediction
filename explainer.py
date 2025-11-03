import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class LIMEExplainer:
    def __init__(self, blackbox_model, feature_names, n_samples=1000, random_state=42):
        self.blackbox_model = blackbox_model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_perturbations(self, instance, X_train):
        n_features = instance.shape[0]
        perturbations = np.zeros((self.n_samples, n_features))

        feature_means = X_train.mean(axis=0).values
        feature_stds = X_train.std(axis=0).values

        for i in range(self.n_samples):
            perturbation = np.zeros(n_features)
            n_features_to_perturb = np.random.randint(1, n_features + 1)
            features_to_perturb = np.random.choice(n_features, n_features_to_perturb, replace=False)

            for j in range(n_features):
                if j in features_to_perturb:
                    noise_scale = np.random.uniform(0.1, 2.0)
                    perturbation[j] = np.random.normal(feature_means[j], feature_stds[j] * noise_scale)
                else:
                    perturbation[j] = instance[j]

            perturbations[i] = perturbation

        return perturbations

    def compute_weights(self, instance, perturbations):
        kernel_width = 0.75 * np.sqrt(len(instance))
        instance_reshaped = instance.reshape(1, -1)
        distances = np.sqrt(np.sum((perturbations - instance_reshaped) ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        weights = weights / np.sum(weights)
        return weights

    def explain_instance(self, instance, X_train):
        instance_array = instance.values if isinstance(instance, pd.Series) else instance

        perturbations = self.generate_perturbations(instance_array, X_train)

        blackbox_predictions = self.blackbox_model.predict_proba(perturbations)[:, 1]

        weights = self.compute_weights(instance_array, perturbations)

        surrogate_model = Ridge(alpha=1.0)
        sqrt_weights = np.sqrt(weights)
        X_weighted = perturbations * sqrt_weights[:, np.newaxis]
        y_weighted = blackbox_predictions * sqrt_weights
        surrogate_model.fit(X_weighted, y_weighted)

        coefficients = surrogate_model.coef_

        contributions = coefficients * instance_array
        feature_importance = list(zip(self.feature_names, coefficients, contributions))
        feature_importance_sorted = sorted(feature_importance, key=lambda x: abs(x[2]), reverse=True)

        original_prediction = self.blackbox_model.predict(instance_array.reshape(1, -1))[0]
        original_proba = self.blackbox_model.predict_proba(instance_array.reshape(1, -1))[0]

        return {
            'feature_contributions': feature_importance_sorted,
            'predicted_class': original_prediction,
            'predicted_probabilities': original_proba,
            'coefficients': coefficients
        }


class CounterfactualAnalyzer:
    def __init__(self, model, feature_names, X_train, categorical_features=None):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.categorical_features = categorical_features if categorical_features else []
        self.feature_stats = self._compute_feature_stats()

    def _compute_feature_stats(self):
        stats = {}
        for col in self.X_train.columns:
            stats[col] = {
                'mean': self.X_train[col].mean(),
                'std': self.X_train[col].std(),
                'min': self.X_train[col].min(),
                'max': self.X_train[col].max(),
                'q25': self.X_train[col].quantile(0.25),
                'q75': self.X_train[col].quantile(0.75),
                'unique_values': sorted(self.X_train[col].unique())
            }
        return stats

    def analyze_feature_impact(self, instance, feature_name, num_steps=20):
        instance_df = pd.DataFrame([instance], columns=self.feature_names)
        original_pred = self.model.predict(instance_df)[0]
        original_proba = self.model.predict_proba(instance_df)[0]

        feature_idx = self.feature_names.index(feature_name)
        feature_stat = self.feature_stats[feature_name]

        is_categorical = len(feature_stat['unique_values']) < 8

        if is_categorical:
            feature_values = np.array(feature_stat['unique_values'])
        else:
            min_val = max(feature_stat['min'], instance[feature_idx] - 3 * feature_stat['std'])
            max_val = min(feature_stat['max'], instance[feature_idx] + 3 * feature_stat['std'])
            feature_values = np.linspace(min_val, max_val, num_steps)

        predictions = []
        probabilities = []

        for val in feature_values:
            modified_instance = instance.copy()
            modified_instance[feature_idx] = val
            modified_df = pd.DataFrame([modified_instance], columns=self.feature_names)

            pred = self.model.predict(modified_df)[0]
            proba = self.model.predict_proba(modified_df)[0]

            predictions.append(pred)
            probabilities.append(proba)

        return {
            'feature_name': feature_name,
            'feature_values': feature_values,
            'predictions': predictions,
            'probabilities': np.array(probabilities),
            'original_value': instance[feature_idx],
            'original_prediction': original_pred,
            'original_probability': original_proba,
            'is_categorical': is_categorical
        }

    def find_minimal_change(self, instance, feature_name, target_class, max_iterations=200):
        feature_idx = self.feature_names.index(feature_name)
        feature_stat = self.feature_stats[feature_name]

        current_instance = instance.copy()
        original_value = instance[feature_idx]

        unique_values = feature_stat['unique_values']
        is_categorical = len(unique_values) < 8

        if is_categorical:
            tested_values = []
            for test_value in unique_values:
                if test_value == original_value:
                    continue

                test_instance = current_instance.copy()
                test_instance[feature_idx] = test_value
                test_df = pd.DataFrame([test_instance], columns=self.feature_names)
                pred = self.model.predict(test_df)[0]

                if pred == target_class:
                    change = test_value - original_value
                    tested_values.append({
                        'value': test_value,
                        'change': abs(change)
                    })

            if tested_values:
                best = min(tested_values, key=lambda x: x['change'])
                return {
                    'success': True,
                    'direction': 'change to',
                    'original_value': original_value,
                    'new_value': best['value'],
                    'change': best['value'] - original_value,
                    'is_categorical': True,
                    'tested_values_count': len(unique_values) - 1
                }
            else:
                return {
                    'success': False,
                    'message': f'Tested all {len(unique_values) - 1} possible values for {feature_name}, none flip the prediction',
                    'is_categorical': True,
                    'tested_values_count': len(unique_values) - 1
                }
        else:
            step_size = feature_stat['std'] * 0.05

            test_instance = current_instance.copy()
            for i in range(max_iterations):
                new_value = original_value + (i + 1) * step_size
                if new_value > feature_stat['max']:
                    break

                test_instance[feature_idx] = new_value
                test_df = pd.DataFrame([test_instance], columns=self.feature_names)
                pred = self.model.predict(test_df)[0]

                if pred == target_class:
                    return {
                        'success': True,
                        'direction': 'increase',
                        'original_value': original_value,
                        'new_value': new_value,
                        'change': new_value - original_value,
                        'iterations': i + 1,
                        'is_categorical': False
                    }

            test_instance = current_instance.copy()
            for i in range(max_iterations):
                new_value = original_value - (i + 1) * step_size
                if new_value < feature_stat['min']:
                    break

                test_instance[feature_idx] = new_value
                test_df = pd.DataFrame([test_instance], columns=self.feature_names)
                pred = self.model.predict(test_df)[0]

                if pred == target_class:
                    return {
                        'success': True,
                        'direction': 'decrease',
                        'original_value': original_value,
                        'new_value': new_value,
                        'change': new_value - original_value,
                        'iterations': i + 1,
                        'is_categorical': False
                    }

            return {
                'success': False,
                'message': f'Tested values from {feature_stat["min"]:.2f} to {feature_stat["max"]:.2f} for {feature_name}, none flip the prediction',
                'is_categorical': False,
                'range_tested': (feature_stat['min'], feature_stat['max'])
            }
