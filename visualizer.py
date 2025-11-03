import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800'
        }

    def plot_feature_impact(self, analysis_result, user_provided=True):
        """Plot how feature changes affect prediction"""
        feature_name = analysis_result['feature_name']
        feature_values = analysis_result['feature_values']
        probabilities = analysis_result['probabilities']
        original_value = analysis_result['original_value']
        original_pred = analysis_result['original_prediction']

        fig = go.Figure()
        classes = ['Lived' , 'Died' , 'Euthanized']
        # probability for each class
        for i in range(probabilities.shape[1]):
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=probabilities[:, i],
                mode='lines',
                name=classes[i],
                line=dict(width=2)
            ))

        # vertical line for original value
        if user_provided:
            fig.add_vline(
                x=original_value,
                line_dash="dash",
                line_color="red",
                annotation_text="Your Input",
                annotation_position="top"
            )

        fig.update_layout(
            title=f'Impact of {feature_name} on Prediction',
            xaxis_title=feature_name,
            yaxis_title='Prediction Probability',
            hovermode='x unified',
            height=400,
            showlegend=True,
            template='plotly_white'
        )

        return fig

    def plot_counterfactual_changes(self, counterfactual_results, user_provided_features):
        """Plot required changes for each feature to flip prediction"""
        valid_results = []
        feature_names = []
        changes = []

        for feature, result in counterfactual_results.items():
            if feature in user_provided_features and result['success']:
                valid_results.append(result)
                feature_names.append(feature)
                changes.append(abs(result['change']))

        if not valid_results:
            return None

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=feature_names,
            y=changes,
            marker_color=self.colors['primary'],
            text=[f"{c:.2f}" for c in changes],
            textposition='auto',
        ))

        fig.update_layout(
            title='Minimal Change Required to Flip Prediction',
            xaxis_title='Feature',
            yaxis_title='Absolute Change Required',
            height=400,
            template='plotly_white',
            xaxis_tickangle=-45
        )

        return fig

    def plot_feature_importance(self, feature_importance, feature_names, top_n=10):
        """Plot feature importance from model"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(top_n)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=self.colors['primary'],
            text=[f"{i:.3f}" for i in importance_df['importance']],
            textposition='auto',
        ))

        fig.update_layout(
            title=f'Top {top_n} Most Important Features',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=500,
            template='plotly_white'
        )

        return fig

    def plot_prediction_confidence(self, probabilities, class_names):
        """Plot prediction confidence for each class"""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=[self.colors['success'] if p == max(probabilities)
                         else self.colors['secondary'] for p in probabilities],
            text=[f"{p:.1%}" for p in probabilities],
            textposition='auto',
        ))

        fig.update_layout(
            title='Prediction Confidence by Class',
            xaxis_title='Outcome',
            yaxis_title='Probability',
            yaxis_range=[0, 1],
            height=350,
            template='plotly_white'
        )

        return fig

    def create_summary_metrics(self, prediction, probabilities, confidence):
        """Create summary metrics display with dark text on light background"""
        metrics_html = f"""
        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; flex: 1; margin: 0 10px;">
                <h3 style="margin: 0; color: #1f77b4;">Predicted Outcome</h3>
                <p style="font-size: 32px; font-weight: bold; margin: 10px 0; color: #2c3e50;">{prediction}</p>
            </div>
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; flex: 1; margin: 0 10px;">
                <h3 style="margin: 0; color: #1f77b4;">Confidence</h3>
                <p style="font-size: 32px; font-weight: bold; margin: 10px 0; color: #2c3e50;">{confidence:.1%}</p>
            </div>
        </div>
        """
        return metrics_html
