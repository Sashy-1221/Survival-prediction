import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from data_loader import DataLoader
from model_trainer import ModelTrainer
from explainer import LIMEExplainer, CounterfactualAnalyzer
from visualizer import Visualizer
from config import FEATURE_DESCRIPTIONS, OUTCOME_DESCRIPTIONS, APP_CONFIG

st.set_page_config(
    page_title=APP_CONFIG['title'],
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stAlert {
        margin-top: 20px;
    }
    .feature-input {
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_resource
def load_and_train_model():
    """Load data and train model (cached)"""
    try:
        data_loader = DataLoader('horse.csv')
        data_loader.load_data()
        X, y = data_loader.preprocess_data()
        feature_stats = data_loader.get_feature_stats()

        model_trainer = ModelTrainer(**APP_CONFIG['model_params'])
        train_results = model_trainer.train(X, y)

        return {
            'data_loader': data_loader,
            'model_trainer': model_trainer,
            'train_results': train_results,
            'feature_stats': feature_stats,
            'X': X,
            'y': y
        }
    except Exception as e:
        st.error(f"Error loading data or training model: {str(e)}")
        return None

def create_input_form(feature_names, feature_stats):
    """Create input form for user to enter feature values"""
    st.sidebar.header("Enter Horse Medical Data")
    st.sidebar.markdown("*Leave blank to use default values*")

    user_inputs = {}
    user_provided = {}

    for feature in feature_names:
        if feature == 'outcome':
            continue

        feature_info = FEATURE_DESCRIPTIONS.get(feature, {
            'name': feature,
            'description': 'No description available',
            'type': 'numerical'
        })

        stat = feature_stats[feature]
        default_value = stat['default']

        with st.sidebar.expander(f"**{feature_info['name']}**"):
            st.markdown(f"*{feature_info['description']}*")

            if stat['type'] == 'numerical':
                value = st.number_input(
                    f"Value (default: {default_value:.2f})",
                    value=None,
                    key=feature,
                    help=f"Leave empty to use mean value: {default_value:.2f}"
                )
            else:
                value = st.number_input(
                    f"Value (default: {default_value:.0f})",
                    value=None,
                    key=feature,
                    step=1,
                    help=f"Leave empty to use median value: {default_value:.0f}"
                )

            if value is not None:
                user_inputs[feature] = value
                user_provided[feature] = True
            else:
                user_inputs[feature] = default_value
                user_provided[feature] = False

    return user_inputs, user_provided

def main():
    st.markdown('<p class="main-header">Horse Survival Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict horse survival with explainable AI and counterfactual analysis</p>', unsafe_allow_html=True)

    with st.spinner("Loading data and training model..."):
        model_data = load_and_train_model()

    if model_data is None:
        st.error("Failed to initialize application. Please check the data file.")
        return

    data_loader = model_data['data_loader']
    model_trainer = model_data['model_trainer']
    train_results = model_data['train_results']
    feature_stats = model_data['feature_stats']
    X = model_data['X']
    y = model_data['y']

    feature_names = data_loader.feature_names

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("Training Accuracy", f"{train_results['train_accuracy']:.2%}")
    st.sidebar.metric("Testing Accuracy", f"{train_results['test_accuracy']:.2%}")
    st.sidebar.markdown("---")

    user_inputs, user_provided = create_input_form(feature_names, feature_stats)

    if st.sidebar.button("Make Prediction", type="primary", use_container_width=True):
        input_df = pd.DataFrame([user_inputs], columns=feature_names)
        input_array = input_df.values[0]

        prediction = model_trainer.predict(input_df)[0]
        probabilities = model_trainer.predict_proba(input_df)[0]
        confidence = max(probabilities)

        st.session_state.prediction = prediction
        st.session_state.probabilities = probabilities
        st.session_state.confidence = confidence
        st.session_state.input_df = input_df
        st.session_state.input_array = input_array
        st.session_state.user_provided = user_provided
        st.session_state.prediction_made = True

    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        prediction = st.session_state.prediction
        probabilities = st.session_state.probabilities
        confidence = st.session_state.confidence
        input_df = st.session_state.input_df
        input_array = st.session_state.input_array
        user_provided = st.session_state.user_provided

        # visualizer
        viz = Visualizer()

        st.markdown("## Prediction Results")

        # outcome name
        outcome_name = OUTCOME_DESCRIPTIONS.get(prediction, f"Class {prediction}")

        # Summary metrics
        summary_html = viz.create_summary_metrics(outcome_name, probabilities, confidence)
        st.markdown(summary_html, unsafe_allow_html=True)

        # Prediction confidence chart
        class_names = [OUTCOME_DESCRIPTIONS.get(i, f"Class {i}")
                      for i in range(len(probabilities))]
        confidence_fig = viz.plot_prediction_confidence(probabilities, class_names)
        st.plotly_chart(confidence_fig, use_container_width=True)

        # Feature importance
        st.markdown("## Feature Importance Analysis")
        st.markdown("*Shows which features are most important for the model's predictions overall*")

        feature_importance = model_trainer.get_feature_importance()
        importance_fig = viz.plot_feature_importance(feature_importance, feature_names, top_n=10)
        st.plotly_chart(importance_fig, use_container_width=True)

        # Counterfactual Analysis - for user-provided features
        st.markdown("## Feature Impact Analysis")
        st.markdown("*Shows how changes to YOUR INPUT values would affect the prediction*")

        user_provided_features = [f for f, provided in user_provided.items() if provided]

        if not user_provided_features:
            st.info("You used all default values. Enter at least one feature value to see impact analysis.")
        else:
            st.success(f"Analyzing impact of {len(user_provided_features)} features you provided")

            counterfactual_analyzer = CounterfactualAnalyzer(
                model_trainer.model,
                feature_names,
                train_results['X_train']
            )

            tabs = st.tabs([feat for feat in user_provided_features])

            for idx, feature in enumerate(user_provided_features):
                with tabs[idx]:
                    feature_info = FEATURE_DESCRIPTIONS.get(feature, {
                        'name': feature,
                        'description': 'No description available'
                    })

                    st.markdown(f"### {feature_info['name']}")
                    st.markdown(f"*{feature_info['description']}*")

                    with st.spinner(f"Analyzing {feature}..."):
                        impact_analysis = counterfactual_analyzer.analyze_feature_impact(
                            input_array, feature, num_steps=30
                        )

                    # feature impact
                    impact_fig = viz.plot_feature_impact(impact_analysis, user_provided=True)
                    st.plotly_chart(impact_fig, use_container_width=True)

                    # Show current value and prediction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Your Input", f"{impact_analysis['original_value']:.2f}")
                    with col2:
                        st.metric("Current Prediction",
                                 OUTCOME_DESCRIPTIONS.get(impact_analysis['original_prediction'],
                                                        f"Class {impact_analysis['original_prediction']}"))
                    with col3:
                        st.metric("Confidence",
                                 f"{max(impact_analysis['original_probability']):.1%}")

                    st.markdown("#### Change Required to Flip Prediction")

                    target_class = 1 - prediction if len(np.unique(y)) == 2 else (prediction + 1) % len(np.unique(y))

                    with st.spinner("Finding minimal change..."):
                        minimal_change = counterfactual_analyzer.find_minimal_change(
                            input_array, feature, target_class, max_iterations=100
                        )

                    if minimal_change['success']:
                        is_categorical = minimal_change.get('is_categorical', False)

                        if is_categorical:
                            st.success(
                                f"To change prediction to **{OUTCOME_DESCRIPTIONS.get(target_class, f'Class {target_class}')}**, "
                                f"change **{feature_info['name']}** from **{minimal_change['original_value']:.0f}** "
                                f"to **{minimal_change['new_value']:.0f}**"
                            )
                        else:
                            direction_text = "Increase" if minimal_change['direction'] == 'increase' else "Decrease"
                            st.success(
                                f"{direction_text}: To change prediction to **{OUTCOME_DESCRIPTIONS.get(target_class, f'Class {target_class}')}**, "
                                f"{minimal_change['direction']} **{feature_info['name']}** by **{abs(minimal_change['change']):.2f}** "
                                f"(from {minimal_change['original_value']:.2f} to {minimal_change['new_value']:.2f})"
                            )
                    else:
                        st.warning(f"{minimal_change.get('message', 'Could not find minimal change')}")

        # Input Summary
        st.markdown("## Input Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### User-Provided Values")
            if user_provided_features:
                provided_data = []
                for feat in user_provided_features:
                    feat_info = FEATURE_DESCRIPTIONS.get(feat, {'name': feat})
                    provided_data.append({
                        'Feature': feat_info['name'],
                        'Value': f"{user_inputs[feat]:.2f}" if isinstance(user_inputs[feat], float) else str(user_inputs[feat])
                    })
                st.dataframe(pd.DataFrame(provided_data), use_container_width=True, hide_index=True)
            else:
                st.info("No values provided by user")

        with col2:
            st.markdown("### Default Values Used")
            default_features = [f for f, provided in user_provided.items() if not provided]
            if default_features:
                default_data = []
                for feat in default_features[:10]:
                    feat_info = FEATURE_DESCRIPTIONS.get(feat, {'name': feat})
                    default_data.append({
                        'Feature': feat_info['name'],
                        'Value': f"{user_inputs[feat]:.2f}" if isinstance(user_inputs[feat], float) else str(user_inputs[feat])
                    })
                st.dataframe(pd.DataFrame(default_data), use_container_width=True, hide_index=True)
                if len(default_features) > 10:
                    st.caption(f"... and {len(default_features) - 10} more")
            else:
                st.info("All values provided by user")

if __name__ == "__main__":
    main()
