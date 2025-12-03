"""
F1 Race Outcome Predictor - Streamlit Dashboard
Provides two modes:
1. Past Race Analysis - Compare predictions vs actual results
2. Future Race Prediction - Predict upcoming races
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

from config import (
    STREAMLIT_CONFIG, FEATURE_COLUMNS, POINTS_SYSTEM,
    TEAM_CANONICAL_MAP, ENSEMBLE_WEIGHTS, TRAINING_SEASONS
)
from data_loader import F1DataLoader
from feature_builder import FeatureBuilder
from train_model import load_trained_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(**STREAMLIT_CONFIG)

# ==========================
# LOAD MODEL (CACHED)
# ==========================

@st.cache_resource
def load_model_artifacts():
    """Load trained models once and cache."""
    try:
        artifacts = load_trained_model('f1_model_artifacts.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("❌ Model artifacts not found! Please run `python train_model.py` first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


@st.cache_resource
def get_data_loader():
    """Create data loader instance (cached)."""
    return F1DataLoader()


# ==========================
# UTILITY FUNCTIONS
# ==========================

def calculate_points(position: int) -> int:
    """Convert position to championship points."""
    return POINTS_SYSTEM.get(int(position), 0)


def predict_race(features_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Generate predictions for a race.
    
    Args:
        features_df: DataFrame with FEATURE_COLUMNS
        artifacts: Loaded model artifacts
    
    Returns:
        DataFrame with predictions from all models
    """
    models = artifacts['models']
    imputer = artifacts['imputer']
    
    X = features_df[FEATURE_COLUMNS]
    X_imputed = imputer.transform(X)
    
    predictions = features_df[['DriverCode', 'TeamName']].copy()
    
    # Individual model predictions
    if 'rf' in models:
        predictions['Pred_RF'] = np.clip(models['rf'].predict(X_imputed), 1, 20)
    if 'xgb' in models:
        predictions['Pred_XGB'] = np.clip(models['xgb'].predict(X_imputed), 1, 20)
    if 'lr' in models:
        predictions['Pred_LR'] = np.clip(models['lr'].predict(X_imputed), 1, 20)
    
    # Ensemble prediction
    if 'xgb' in models:
        predictions['Ensemble_Pred'] = (
            ENSEMBLE_WEIGHTS['rf'] * predictions['Pred_RF'] +
            ENSEMBLE_WEIGHTS['xgb'] * predictions['Pred_XGB'] +
            ENSEMBLE_WEIGHTS['lr'] * predictions['Pred_LR']
        )
    else:
        predictions['Ensemble_Pred'] = (
            0.5 * predictions['Pred_RF'] +
            0.5 * predictions['Pred_LR']
        )
    
    predictions['Ensemble_Pred'] = np.clip(predictions['Ensemble_Pred'], 1, 20)
    
    return predictions


# ==========================
# UI COMPONENTS
# ==========================

def render_header():
    """Render page header."""
    st.title("🏎️ F1 Race Outcome Predictor")
    st.markdown("**Machine Learning-powered race predictions for Formula 1**")
    st.markdown("---")


def render_mode_selector() -> str:
    """Render mode selection radio buttons."""
    mode = st.radio(
        "Select Mode:",
        options=["Past Race Analysis", "Future Race Prediction"],
        horizontal=True,
        help="Past: Validate predictions against actual results. Future: Predict upcoming races."
    )
    return mode


# ==========================
# MODE 1: PAST RACE ANALYSIS
# ==========================

def render_past_race_analysis(artifacts: dict, loader: F1DataLoader):
    """Render Past Race Analysis mode."""
    
    st.header("📊 Past Race Analysis")
    st.markdown("Compare model predictions against actual race results.")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Race Selection")
        
        # Season selector
        season = st.selectbox(
            "Season",
            options=TRAINING_SEASONS,
            index=len(TRAINING_SEASONS) - 1,  # Default to most recent
            help="Select a season from the training data"
        )
        
        # Get completed events for this season
        completed_events = loader.get_completed_events(season)
        
        if not completed_events:
            st.warning(f"No completed events found for {season}")
            return
        
        # Event selector
        event = st.selectbox(
            "Grand Prix",
            options=completed_events,
            help="Select a completed race"
        )
        
        st.markdown("---")
        
        # Model selection
        st.subheader("Model Selection")
        use_rf = st.checkbox("Random Forest", value=True)
        use_xgb = st.checkbox("XGBoost", value='xgb' in artifacts['models'])
        use_lr = st.checkbox("Linear Regression", value=True)
        use_ensemble = st.checkbox("Ensemble", value=True)
        
        st.markdown("---")
        
        analyze_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if analyze_button:
        with st.spinner(f"Loading {season} {event}..."):
            # Load race session
            session = loader.load_race_session(season, event)
            
            if session is None:
                st.error(f"❌ Could not load race data for {season} {event}")
                st.info("This may be due to:")
                st.markdown("- FastF1 API issues\n- Missing data for this session\n- Recent race not yet available")
                return
            
            # Extract results
            race_results = loader.extract_race_results(session)
            if race_results is None:
                st.error("❌ Could not extract race results")
                return
            
            # Add metadata
            circuit_info = loader.get_circuit_info(session)
            race_results['Year'] = season
            race_results['EventName'] = circuit_info['event_name']
            race_results['CircuitName'] = circuit_info['circuit_name']
            race_results['RoundNumber'] = circuit_info['round_number']
            race_results['IsWetRace'] = int(loader.is_wet_race(session))
            race_results['TeamCanonical'] = race_results['TeamName'].map(
                lambda x: TEAM_CANONICAL_MAP.get(x, x)
            )
            
            # Build features (using historical stats from training)
            fb_stats = artifacts['feature_builder']
            temp_fb = FeatureBuilder(pd.DataFrame())  # Empty base data
            temp_fb.driver_stats = fb_stats['driver_stats']
            temp_fb.team_stats = fb_stats['team_stats']
            temp_fb.circuit_stats = fb_stats['circuit_stats']
            temp_fb.driver_map = fb_stats['driver_map']
            temp_fb.team_map = fb_stats['team_map']
            temp_fb.circuit_map = fb_stats['circuit_map']
            
            features = temp_fb.build_features(race_results)
            
            # Generate predictions
            predictions = predict_race(features, artifacts)
            
            # Merge with actual results
            results_df = race_results.copy()
            results_df = results_df.merge(predictions, on=['DriverCode', 'TeamName'], how='left')
            results_df = results_df.dropna(subset=['Position'])
            results_df = results_df.sort_values('Position')
            
            # Calculate metrics
            results_df['AbsError'] = np.abs(results_df['Position'] - results_df['Ensemble_Pred'])
            results_df['PointsActual'] = results_df['Position'].apply(calculate_points)
            results_df['PointsPredicted'] = results_df['Ensemble_Pred'].apply(lambda x: calculate_points(round(x)))
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mae = results_df['AbsError'].mean()
                st.metric("Mean Absolute Error", f"{mae:.2f} positions")
            
            with col2:
                top3_actual = set(results_df.nsmallest(3, 'Position')['DriverCode'])
                top3_pred = set(results_df.nsmallest(3, 'Ensemble_Pred')['DriverCode'])
                top3_overlap = len(top3_actual & top3_pred)
                st.metric("Top-3 Accuracy", f"{top3_overlap}/3")
            
            with col3:
                points_mae = np.abs(results_df['PointsActual'] - results_df['PointsPredicted']).mean()
                st.metric("Points Error (MAE)", f"{points_mae:.1f}")
            
            with col4:
                wet_indicator = "💧 Wet" if results_df['IsWetRace'].iloc[0] == 1 else "☀️ Dry"
                st.metric("Conditions", wet_indicator)
            
            st.markdown("---")
            
            # Results table
            st.subheader("📋 Detailed Results")
            
            display_df = results_df[[
                'Position', 'DriverCode', 'TeamName', 'GridPosition',
                'Ensemble_Pred', 'AbsError', 'PointsActual', 'PointsPredicted'
            ]].copy()
            
            if use_rf:
                display_df['Pred_RF'] = results_df['Pred_RF']
            if use_xgb and 'Pred_XGB' in results_df.columns:
                display_df['Pred_XGB'] = results_df['Pred_XGB']
            if use_lr:
                display_df['Pred_LR'] = results_df['Pred_LR']
            
            display_df.columns = display_df.columns.str.replace('_', ' ')
            display_df = display_df.round(2)
            
            # Color code error column with better text visibility
            def highlight_error(val):
                if pd.isna(val):
                    return 'background-color: #ffffff; color: #000000'
                elif val < 2:
                    return 'background-color: #d4edda; color: #000000'
                elif val < 5:
                    return 'background-color: #fff3cd; color: #000000'
                else:
                    return 'background-color: #f8d7da; color: #000000'

            styled_df = display_df.style.applymap(
                highlight_error,
                subset=['AbsError'] if 'AbsError' in display_df.columns else []
            )
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Position Comparison")
            
            # Bar chart: Actual vs Predicted
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Actual Position',
                x=results_df['DriverCode'],
                y=results_df['Position'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Predicted Position',
                x=results_df['DriverCode'],
                y=results_df['Ensemble_Pred'],
                marker_color='orange'
            ))
            
            fig.update_layout(
                xaxis_title="Driver",
                yaxis_title="Position",
                yaxis=dict(autorange="reversed"),  # Position 1 at top
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)


# ==========================
# MODE 2: FUTURE RACE PREDICTION
# ==========================

def render_future_prediction(artifacts: dict, loader: F1DataLoader):
    """Render Future Race Prediction mode."""
    
    st.header("🔮 Future Race Prediction")
    st.markdown("Predict outcomes for upcoming races.")
    
    with st.sidebar:
        st.subheader("Race Selection")
        
        # Season selector (default to current year)
        current_year = datetime.now().year
        season = st.selectbox(
            "Season",
            options=[2024, 2025],
            index=1 if current_year >= 2025 else 0
        )
        
        # Get upcoming events
        upcoming_events = loader.get_upcoming_events(season)
        
        if upcoming_events:
            event = st.selectbox(
                "Upcoming Grand Prix",
                options=upcoming_events,
                help="Select a future race to predict"
            )
        else:
            st.warning("No upcoming events found. Using custom mode.")
            event = st.text_input("Event Name", "Custom Grand Prix")
        
        # Circuit selection (for stats lookup)
        circuit_name = st.text_input(
            "Circuit Name",
            value="Circuit de Barcelona-Catalunya",
            help="Enter exact circuit name for stats lookup"
        )
        
        st.markdown("---")
        
        # Race conditions
        st.subheader("Race Conditions")
        is_wet = st.radio("Weather", options=["Dry", "Wet"], index=0)
        is_wet_int = 1 if is_wet == "Wet" else 0
        
        st.markdown("---")
        
        predict_button = st.button("🎯 Predict Race", type="primary", use_container_width=True)
    
    # Main content: Grid setup
    st.subheader("🏁 Starting Grid Setup")
    st.markdown("Enter drivers and their qualifying positions (or leave blank for midfield default)")
    
    # Get recent driver list from training data
    fb_stats = artifacts['feature_builder']
    known_drivers = fb_stats['driver_stats']['DriverCode'].tolist()
    known_teams = fb_stats['team_stats']['TeamCanonical'].tolist()
    
    # Create editable grid
    if 'grid_data' not in st.session_state:
        # Initialize with top 10 drivers from last season
        top_drivers = fb_stats['driver_stats'].nsmallest(10, 'DriverAvgPos')['DriverCode'].tolist()
        st.session_state.grid_data = pd.DataFrame({
            'DriverCode': top_drivers + [''] * 10,
            'TeamName': [''] * 20,
            'GridPosition': list(range(1, 21))
        })
    
    # Editable data editor
    edited_grid = st.data_editor(
        st.session_state.grid_data,
        num_rows="fixed",
        column_config={
            "DriverCode": st.column_config.SelectboxColumn(
                "Driver",
                options=known_drivers,
                required=True
            ),
            "TeamName": st.column_config.SelectboxColumn(
                "Team",
                options=known_teams,
                required=True
            ),
            "GridPosition": st.column_config.NumberColumn(
                "Grid Position",
                min_value=1,
                max_value=20,
                step=1
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Save edited grid
    st.session_state.grid_data = edited_grid
    
    # Run prediction
    if predict_button:
        # Validate input
        valid_rows = edited_grid[
            (edited_grid['DriverCode'] != '') & 
            (edited_grid['TeamName'] != '')
        ]
        
        if len(valid_rows) < 5:
            st.error("❌ Please enter at least 5 drivers")
            return
        
        with st.spinner("Generating predictions..."):
            # Prepare prediction data
            prediction_data = valid_rows.copy()
            prediction_data['Year'] = season
            prediction_data['EventName'] = event
            prediction_data['CircuitName'] = circuit_name
            prediction_data['RoundNumber'] = 1  # Approximate
            prediction_data['IsWetRace'] = is_wet_int
            prediction_data['TeamCanonical'] = prediction_data['TeamName']
            
            # Build features
            temp_fb = FeatureBuilder(pd.DataFrame())
            temp_fb.driver_stats = fb_stats['driver_stats']
            temp_fb.team_stats = fb_stats['team_stats']
            temp_fb.circuit_stats = fb_stats['circuit_stats']
            temp_fb.driver_map = fb_stats['driver_map']
            temp_fb.team_map = fb_stats['team_map']
            temp_fb.circuit_map = fb_stats['circuit_map']
            
            features = temp_fb.build_features(prediction_data)
            
            # Generate predictions
            predictions = predict_race(features, artifacts)
            
            # Merge and sort by predicted position
            result = prediction_data.merge(predictions, on=['DriverCode', 'TeamName'])
            result = result.sort_values('Ensemble_Pred')
            result['PredictedPosition'] = range(1, len(result) + 1)
            result['Delta'] = result['GridPosition'] - result['Ensemble_Pred']
            
            # Display results
            st.markdown("---")
            st.subheader("🏆 Predicted Classification")
            
            display_cols = [
                'PredictedPosition', 'DriverCode', 'TeamName', 
                'GridPosition', 'Ensemble_Pred', 'Delta'
            ]
            
            if 'Pred_RF' in result.columns:
                display_cols.append('Pred_RF')
            if 'Pred_XGB' in result.columns:
                display_cols.append('Pred_XGB')
            if 'Pred_LR' in result.columns:
                display_cols.append('Pred_LR')
            
            display_result = result[display_cols].round(2)
            display_result.columns = display_result.columns.str.replace('_', ' ')
            
            st.dataframe(display_result, use_container_width=True, height=600)
            
            # Visualization: Expected position changes
            st.markdown("---")
            st.subheader("📊 Expected Position Changes")
            
            fig = go.Figure()
            
            colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in result['Delta']]
            
            fig.add_trace(go.Bar(
                x=result['DriverCode'],
                y=result['Delta'],
                marker_color=colors,
                text=result['Delta'].round(1),
                textposition='outside'
            ))
            
            fig.update_layout(
                xaxis_title="Driver",
                yaxis_title="Expected Gain/Loss (Positions)",
                height=400
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Edge case warnings
            st.markdown("---")
            st.info("ℹ️ **Note**: Predictions use historical driver/team stats. Rookie drivers use team-based estimates.")


# ==========================
# MAIN APP
# ==========================

def main():
    """Main application entry point."""
    
    # Load artifacts
    artifacts = load_model_artifacts()
    loader = get_data_loader()
    
    # Render header
    render_header()
    
    # Mode selection
    mode = render_mode_selector()
    
    st.markdown("---")
    
    # Render appropriate mode
    if mode == "Past Race Analysis":
        render_past_race_analysis(artifacts, loader)
    else:
        render_future_prediction(artifacts, loader)
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About This Model"):
        st.markdown(f"""
        **Training Data**: {artifacts['training_info']['training_seasons']}
        
        **Models Used**:
        - Random Forest Regressor (40% weight)
        - {'XGBoost Regressor (40% weight)' if artifacts['training_info']['xgboost_available'] else 'XGBoost unavailable'}
        - Linear Regression (20% weight)
        
        **Features**: Driver stats, team stats, circuit stats, grid position, weather, engineered features
        
        **Limitations**:
        - Historical data only (no real-time updates)
        - Rookie drivers use team-based estimates
        - New circuits use fallback statistics
        - Weather is simplified (wet/dry only)
        
        **Data Source**: FastF1 (unofficial F1 timing data)
        """)


if __name__ == '__main__':
    main()