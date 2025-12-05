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
from typing import Dict

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

def get_current_season_drivers_teams(loader: F1DataLoader, year: int = None) -> Dict[str, str]:
    """
    Get driver-team combinations for a specific season.
    Returns a dictionary mapping driver codes to team names.
    """
    if year is None:
        year = datetime.now().year

    try:
        # Try to get the season's driver-team info from FastF1
        # For future seasons like 2025, this may not work, so we'll provide fallbacks
        import fastf1

        # Try to get data from the first race of the season
        try:
            race = fastf1.get_session(year, 1, 'R')  # First race of the season
            if race and hasattr(race, 'drivers'):
                # Load the race to get driver info
                race.load(laps=False, telemetry=False, weather=False, drivers=True)
                driver_team_map = {}

                # Get driver-team combinations
                for drv_num in race.drivers:
                    try:
                        drv = race.get_driver(drv_num)
                        if drv.code and drv.team_name:
                            driver_team_map[drv.code] = drv.team_name
                    except:
                        continue

                return driver_team_map
        except Exception as e:
            logger.warning(f"Could not fetch {year} drivers via FastF1: {e}")

    except Exception as e:
        logger.warning(f"Could not fetch {year} drivers: {e}")

    # For future seasons (like 2025), use confirmed driver roster
    # This is our fallback for seasons where FastF1 data is not available
    if year == 2025:
        # 2025 confirmed driver roster based on official announcements
        return {
            'VER': 'Red Bull',          # Max Verstappen
            'LAW': 'Red Bull',          # Liam Lawson
            'LEC': 'Ferrari',           # Charles Leclerc
            'HAM': 'Ferrari',           # Lewis Hamilton
            'NOR': 'McLaren',           # Lando Norris
            'PIA': 'McLaren',           # Oscar Piastri
            'RUS': 'Mercedes',          # George Russell
            'ANT': 'Mercedes',          # Andrea Kimi Antonelli (new driver)
            'ALO': 'Aston Martin',      # Fernando Alonso
            'STR': 'Aston Martin',      # Lance Stroll
            'TSU': 'RB',                # Yuki Tsunoda
            'HAD': 'RB',                # Isack Hadjar (new driver)
            'BEA': 'Haas',              # Oliver Bearman (new driver)
            'OCO': 'Haas',              # Esteban Ocon
            'GAS': 'Alpine',            # Pierre Gasly
            'DOO': 'Alpine',            # Jack Doohan (new driver)
            'DEV': 'Williams',          # Alex Albon
            'SAI': 'Williams',          # Carlos Sainz
            'HUL': 'Kick Sauber',       # Nico Hulkenberg
            'BOR': 'Kick Sauber',       # Gabriel Bortoleto (new driver)
        }

    # Fallback to most recent training data for other cases
    return {}


def get_driver_names() -> Dict[str, str]:
    """
    Get full driver names for driver codes.
    Returns a dictionary mapping driver codes to full names.
    """
    return {
        'VER': 'Max Verstappen',          # Red Bull
        'LAW': 'Liam Lawson',             # Red Bull
        'LEC': 'Charles Leclerc',         # Ferrari
        'HAM': 'Lewis Hamilton',          # Ferrari
        'NOR': 'Lando Norris',            # McLaren
        'PIA': 'Oscar Piastri',           # McLaren
        'RUS': 'George Russell',          # Mercedes
        'ANT': 'Andrea Kimi Antonelli',   # Mercedes (new driver)
        'ALO': 'Fernando Alonso',         # Aston Martin
        'STR': 'Lance Stroll',            # Aston Martin
        'TSU': 'Yuki Tsunoda',            # RB
        'HAD': 'Isack Hadjar',           # RB (new driver)
        'BEA': 'Oliver Bearman',         # Haas (new driver)
        'OCO': 'Esteban Ocon',            # Haas
        'GAS': 'Pierre Gasly',            # Alpine
        'DOO': 'Jack Doohan',            # Alpine (new driver)
        'DEV': 'Alex Albon',              # Williams
        'SAI': 'Carlos Sainz',            # Williams
        'HUL': 'Nico Hulkenberg',        # Kick Sauber
        'BOR': 'Gabriel Bortoleto',      # Kick Sauber (new driver)
    }


def explain_prediction_with_shap(driver_code: str, team_name: str, features_row: pd.Series, artifacts: dict, prediction: float) -> str:
    """
    Generate a human-readable explanation for why a driver was predicted to finish in a certain position using SHAP.

    Args:
        driver_code: Driver code (e.g., 'VER', 'HAM')
        team_name: Team name
        features_row: Feature values for this driver (as a Series)
        artifacts: Loaded model artifacts (including SHAP explainer)
        prediction: Predicted finishing position

    Returns:
        String explanation of prediction factors based on SHAP values
    """
    driver_names = get_driver_names()
    driver_name = driver_names.get(driver_code, driver_code)

    # Check if SHAP is available
    if not artifacts['training_info']['shap_available'] or 'shap_explainer' not in artifacts or artifacts['shap_explainer'] is None:
        # Fallback to basic explanation if SHAP is not available
        return f"{driver_name} ({driver_code}) prediction based on historical data and team performance."

    try:
        # Get the trained model (Random Forest) to use for SHAP
        rf_model = artifacts['models']['rf']
        shap_explainer = artifacts['shap_explainer']

        # Get the feature values (reshaped for single prediction)
        feature_values = features_row[FEATURE_COLUMNS].values.reshape(1, -1)

        # Transform features using the imputer (as done during training)
        imputed_features = artifacts['imputer'].transform(feature_values)

        # Get SHAP values for this prediction
        shap_values = shap_explainer.shap_values(imputed_features)

        # Get the base value (expected prediction without any features)
        base_value = shap_explainer.expected_value

        # Create feature importance list to identify top contributing factors
        feature_names = FEATURE_COLUMNS
        shap_importance = [(feature_names[i], shap_values[0][i]) for i in range(len(feature_names))]

        # Sort by absolute SHAP value to get most impactful features
        shap_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        # Get the top 2 most impactful features
        top_features = shap_importance[:2]

        explanations = []
        for feature_name, shap_val in top_features:
            # Format based on the feature type
            feature_val = features_row[feature_name]

            if feature_name == 'GridPosition':
                if shap_val < 0:  # Negative SHAP value means better prediction (lower position number)
                    explanations.append(f"Starting from pole position ({int(feature_val)}) gave significant advantage")
                else:
                    explanations.append(f"Starting from back grid ({int(feature_val)}) was disadvantageous")
            elif feature_name == 'DriverAvgPos':
                if shap_val < 0:
                    explanations.append(f"Strong historical average finish position ({feature_val:.1f})")
                else:
                    explanations.append(f"Weaker historical average finish position ({feature_val:.1f})")
            elif feature_name == 'TeamAvgPos':
                if shap_val < 0:
                    explanations.append(f"Competitive team with strong historical performance ({feature_val:.1f})")
                else:
                    explanations.append(f"Less competitive team with historical performance ({feature_val:.1f})")
            elif feature_name == 'DriverDNFRate':
                if shap_val > 0:
                    explanations.append(f"Higher DNF rate ({feature_val:.2f}) affected reliability expectations")
                else:
                    explanations.append(f"Good reliability record ({feature_val:.2f}) supported confidence")
            elif feature_name == 'TeamDNFRate':
                if shap_val > 0:
                    explanations.append(f"Team reliability concerns ({feature_val:.2f} DNF rate)")
                else:
                    explanations.append(f"Team reliability strength ({feature_val:.2f} DNF rate)")
            elif feature_name == 'IsWetRace' and feature_val == 1:
                explanations.append("Wet weather conditions expected to impact performance")
            else:
                # General description for other features
                direction = "positively" if shap_val < 0 else "negatively"
                explanations.append(f"{feature_name} value of {feature_val:.2f} affected prediction {direction}")

        if explanations:
            explanation_text = f"{driver_name} ({driver_code}) driving for {team_name}: " + "; ".join(explanations) + "."
            return explanation_text
        else:
            return f"{driver_name} ({driver_code}) prediction based on historical data and team performance."

    except Exception as e:
        # If SHAP explanation fails for any reason, return a basic explanation
        logger.warning(f"SHAP explanation failed for {driver_code}: {e}")
        return f"{driver_name} ({driver_code}) prediction based on historical data and team performance."


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
            st.subheader("📈 Prediction Accuracy")

            # Proper scatter plot for prediction error analysis
            fig = go.Figure()

            # Add perfect prediction line (y=x)
            max_pos = max(results_df['Position'].max(), results_df['Ensemble_Pred'].max()) + 1
            fig.add_trace(go.Scatter(
                x=[1, max_pos],
                y=[1, max_pos],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction',
                showlegend=True
            ))

            # Add actual vs predicted points with color coding based on error magnitude
            errors = np.abs(results_df['Position'] - results_df['Ensemble_Pred'])
            max_error = errors.max() if len(errors) > 0 else 1

            fig.add_trace(go.Scatter(
                x=results_df['Position'],
                y=results_df['Ensemble_Pred'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=errors,
                    colorscale=[
                        [0, 'green'],      # Low error (good prediction)
                        [0.5, 'yellow'],   # Medium error
                        [1, 'red']         # High error (poor prediction)
                    ],
                    showscale=True,
                    colorbar=dict(title="Prediction Error"),
                    opacity=0.8
                ),
                text=results_df['DriverCode'] + '<br>Actual: ' + results_df['Position'].astype(str) +
                     '<br>Pred: ' + results_df['Ensemble_Pred'].round(1).astype(str) +
                     '<br>Error: ' + errors.round(1).astype(str),
                hovertemplate='%{text}<extra></extra>',
                name='Driver Predictions'
            ))

            fig.update_layout(
                title="Actual vs Predicted Positions (Perfect = Points on Red Line)",
                xaxis_title="Actual Finishing Position (1st at left, worse to right)",
                yaxis_title="Predicted Finishing Position (1st at bottom, worse up)",
                xaxis=dict(autorange="reversed", dtick=1, range=[max_pos, 0.5]),
                yaxis=dict(autorange="reversed", dtick=1, range=[max_pos, 0.5]),
                width=700,
                height=600,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **How to read**: Points on red line = perfect prediction. Green points = low error, Red points = high error. Closer to line = better prediction.")


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
        # Use a more common default or allow selection from known circuits
        upcoming_circuits = [
            "Bahrain International Circuit", "Circuit de Monaco", "Circuit Gilles Villeneuve",
            "Red Bull Ring", "Silverstone", "Hungaroring", "Spa-Francorchamps", "Zandvoort",
            "Monza", "Baku City Circuit", "Circuit of the Americas", "Autódromo Hermanos Rodríguez",
            "Interlagos", "Las Vegas Strip Circuit", "Yas Marina Circuit", "Losail International Circuit"
        ]

        circuit_name = st.selectbox(
            "Circuit",
            options=upcoming_circuits,
            index=upcoming_circuits.index("Yas Marina Circuit"),  # Default to Yas Marina Circuit
            help="Select the circuit for the race"
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
    st.markdown("Select drivers and their qualifying positions (teams will be auto-populated when you select a driver)")
    st.info("💡 **Tip**: Use the 'Update Grid' button after making changes to save your selections.")

    # Demo mode button
    if st.button("🎯 Load Demo Grid (2025 Season)", type="secondary"):
        # Pre-fill with realistic demo data based on 2025 season knowledge
        demo_drivers_teams = {
            0: {'driver_code': 'VER', 'team_name': 'Red Bull', 'grid_position': 1},  # Max Verstappen
            1: {'driver_code': 'LEC', 'team_name': 'Ferrari', 'grid_position': 2},  # Charles Leclerc
            2: {'driver_code': 'HAM', 'team_name': 'Ferrari', 'grid_position': 3},  # Lewis Hamilton
            3: {'driver_code': 'RUS', 'team_name': 'Mercedes', 'grid_position': 4},  # George Russell
            4: {'driver_code': 'NOR', 'team_name': 'McLaren', 'grid_position': 5},  # Lando Norris
            5: {'driver_code': 'PIA', 'team_name': 'McLaren', 'grid_position': 6},  # Oscar Piastri
            6: {'driver_code': 'ANT', 'team_name': 'Mercedes', 'grid_position': 7},  # Kimi Antonelli
            7: {'driver_code': 'ALO', 'team_name': 'Aston Martin', 'grid_position': 8},  # Fernando Alonso
            8: {'driver_code': 'SAI', 'team_name': 'Williams', 'grid_position': 9},  # Carlos Sainz
            9: {'driver_code': 'DEV', 'team_name': 'Williams', 'grid_position': 10}, # Alex Albon
        }

        # Fill the session state with demo data
        for i in range(20):
            if i in demo_drivers_teams:
                st.session_state.driver_grid_rows[i] = demo_drivers_teams[i]
            else:
                st.session_state.driver_grid_rows[i] = {
                    'driver_code': '',
                    'team_name': '',
                    'grid_position': 0
                }

        st.success("Demo grid loaded! Use 'Update Grid' to see the changes.")
        st.rerun()

    # Get driver-team combinations for the selected season (e.g., 2025)
    current_drivers_teams = get_current_season_drivers_teams(loader, season)

    if not current_drivers_teams:
        # Fallback to recent training data
        fb_stats = artifacts['feature_builder']
        # Get all known drivers and teams from training data
        temp_df = pd.DataFrame({
            'DriverCode': fb_stats['driver_stats']['DriverCode'].tolist(),
            'TeamName': [''] * len(fb_stats['driver_stats'])  # Placeholder teams
        })
        # We'll map drivers to their most common team from training data
        # For simplicity, we'll use the team mapping from training data
        for idx, row in fb_stats['driver_stats'].iterrows():
            driver_code = row['DriverCode']
            # Find the team for this driver from raw training data if possible
            # For now, we'll just use the training data teams
    else:
        # We have current season data
        temp_df = pd.DataFrame(list(current_drivers_teams.items()), columns=['DriverCode', 'TeamName'])
        temp_df['GridPosition'] = range(1, len(temp_df) + 1)

    # Create editable grid
    if 'grid_data' not in st.session_state:
        # Initialize with drivers from most recent season
        if current_drivers_teams:
            # Use current season drivers
            drivers = list(current_drivers_teams.keys())
            teams = list(current_drivers_teams.values())
        else:
            # Fallback to drivers from training data
            fb_stats = artifacts['feature_builder']
            drivers = fb_stats['driver_stats'].nsmallest(20, 'DriverAvgPos')['DriverCode'].tolist()
            teams = [''] * len(drivers)  # Teams will be selected

        st.session_state.grid_data = pd.DataFrame({
            'DriverCode': drivers + [''] * max(0, 20 - len(drivers)),
            'TeamName': teams + [''] * max(0, 20 - len(teams)),
            'GridPosition': list(range(1, min(len(drivers) + 1, 21))) + [i for i in range(len(drivers) + 1, 21)],
        })

    # Get known drivers and teams for the selectboxes
    fb_stats = artifacts['feature_builder']
    known_drivers = fb_stats['driver_stats']['DriverCode'].tolist()
    known_teams = fb_stats['team_stats']['TeamCanonical'].tolist()

    # If we have current season data, prioritize these in the selectboxes
    if current_drivers_teams:
        # Update known drivers and teams with current season data
        current_drivers = list(current_drivers_teams.keys())
        current_teams = list(set(current_drivers_teams.values()))
        # Add any current data to the known lists if not already present
        for drv in current_drivers:
            if drv not in known_drivers:
                known_drivers.append(drv)
        for team in current_teams:
            if team not in known_teams:
                known_teams.append(team)

    # Create individual input fields for each row to prevent auto-refresh issues
    # Initialize session state for driver grid if not exists
    if 'driver_grid_rows' not in st.session_state:
        st.session_state.driver_grid_rows = []
        # Initialize with default values for 20 rows
        for i in range(20):
            st.session_state.driver_grid_rows.append({
                'driver_code': '',
                'team_name': '',
                'grid_position': 0  # Changed from i+1 to 0 to indicate no position set yet
            })

    # Create two columns: one for driver reference and one for grid setup
    left_col, right_col = st.columns([1, 2])

    # Left column: Driver reference panel
    with left_col:
        st.subheader("📋 Driver Reference")
        st.markdown("Driver codes and full names")

        # Get driver names
        driver_names = get_driver_names()

        # Create a table of drivers
        if current_drivers_teams:
            # Create a DataFrame for the reference table
            ref_data = []
            for code, team in current_drivers_teams.items():
                full_name = driver_names.get(code, "Unknown Driver")
                ref_data.append({'Code': code, 'Name': full_name, 'Team': team})

            ref_df = pd.DataFrame(ref_data)
            ref_df = ref_df.sort_values('Code')  # Sort alphabetically by code
            st.dataframe(ref_df, use_container_width=True, hide_index=True)
        else:
            st.info("No driver data available")

    # Right column: Grid setup form
    with right_col:
        st.subheader("🏁 Grid Setup")

        # Create a form to group all inputs together
        with st.form(key="driver_grid_form"):
            # Create columns for headers
            header_cols = st.columns([1, 2, 2])
            with header_cols[0]:
                st.markdown("**Grid**")
            with header_cols[1]:
                st.markdown("**Driver**")
            with header_cols[2]:
                st.markdown("**Team**")

            # Create input fields for each row
            for i in range(20):
                row_cols = st.columns([1, 2, 2])

                # Grid position
                with row_cols[0]:
                    position_key = f"position_{i}"
                    current_position = st.session_state.driver_grid_rows[i]['grid_position']

                    selected_position = st.number_input(
                        f"Grid {i+1}",
                        min_value=0,
                        max_value=20,
                        value=int(current_position) if current_position > 0 else 0,
                        step=1,
                        key=position_key,
                        label_visibility="collapsed",
                        help="Set to 0 to skip this position"
                    )

                    # Update session state if position changed
                    if selected_position != st.session_state.driver_grid_rows[i]['grid_position']:
                        st.session_state.driver_grid_rows[i]['grid_position'] = selected_position

                # Driver selection
                with row_cols[1]:
                    driver_key = f"driver_{i}"
                    current_driver = st.session_state.driver_grid_rows[i]['driver_code']

                    selected_driver = st.selectbox(
                        f"Driver {i+1}",
                        options=[''] + list(current_drivers_teams.keys()) if current_drivers_teams else known_drivers,
                        index=0 if not current_driver else (list(current_drivers_teams.keys()).index(current_driver) + 1 if current_driver in current_drivers_teams else 0),
                        key=driver_key,
                        label_visibility="collapsed",
                        format_func=lambda x: f"{driver_names.get(x, x)} ({x})" if x else "Select Driver"
                    )

                    # Update session state if driver changed
                    if selected_driver != st.session_state.driver_grid_rows[i]['driver_code']:
                        st.session_state.driver_grid_rows[i]['driver_code'] = selected_driver
                        # Auto-populate team based on selected driver if team is empty
                        if selected_driver and current_drivers_teams and selected_driver in current_drivers_teams:
                            if not st.session_state.driver_grid_rows[i]['team_name']:  # Only if team is empty
                                st.session_state.driver_grid_rows[i]['team_name'] = current_drivers_teams[selected_driver]

                # Team selection
                with row_cols[2]:
                    team_key = f"team_{i}"
                    current_team = st.session_state.driver_grid_rows[i]['team_name']

                    # Auto-populate team if driver is selected and team is empty
                    selected_driver = st.session_state.driver_grid_rows[i]['driver_code']
                    if (selected_driver and not current_team and
                        current_drivers_teams and selected_driver in current_drivers_teams):
                        current_team = current_drivers_teams[selected_driver]
                        st.session_state.driver_grid_rows[i]['team_name'] = current_team

                    selected_team = st.selectbox(
                        f"Team {i+1}",
                        options=[''] + known_teams,
                        index=0 if not current_team else (known_teams.index(current_team) + 1 if current_team in known_teams else 0),
                        key=team_key,
                        label_visibility="collapsed"
                    )

                    # Update session state if team changed (this preserves user changes)
                    if selected_team != st.session_state.driver_grid_rows[i]['team_name']:
                        st.session_state.driver_grid_rows[i]['team_name'] = selected_team

            # Submit button for the form
            form_submit = st.form_submit_button("Update Grid", type="secondary", use_container_width=True)

    # Convert session state back to DataFrame format for further processing
    grid_data_list = []
    for i, row in enumerate(st.session_state.driver_grid_rows):
        # Only include rows that have a driver selected and a valid grid position
        if row['driver_code'] and row['grid_position'] > 0:
            grid_data_list.append({
                'DriverCode': row['driver_code'],
                'TeamName': row['team_name'],
                'GridPosition': row['grid_position']
            })

    edited_grid = pd.DataFrame(grid_data_list)

    # Sort by grid position to ensure proper order
    if not edited_grid.empty:
        edited_grid = edited_grid.sort_values('GridPosition')

    # Run prediction
    if predict_button:
        # Validate input - check if DataFrame has required columns
        if edited_grid.empty or 'DriverCode' not in edited_grid.columns or 'TeamName' not in edited_grid.columns:
            st.error("❌ Please set up at least 5 drivers with grid positions")
            return

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

            # Add prediction explanations using SHAP
            st.markdown("---")
            st.subheader("🔍 Prediction Explanations")
            st.info("ℹ️ **How this works**: SHAP explains AI predictions by showing which factors most influenced each driver's predicted position.")

            st.markdown("Why each driver was predicted to finish in their position:")

            # Create explanations for each driver using SHAP
            for idx, row in result.iterrows():
                explanation = explain_prediction_with_shap(
                    row['DriverCode'],
                    row['TeamName'],
                    features.iloc[idx],  # Pass the feature row for this driver
                    artifacts,  # Pass the model artifacts containing SHAP explainer
                    row['Ensemble_Pred']
                )
                # Add an emoji based on predicted position
                if row['Ensemble_Pred'] <= 3:
                    emoji = "🏆"
                elif row['Ensemble_Pred'] <= 10:
                    emoji = "🏁"
                else:
                    emoji = "🚗"

                st.markdown(f"{emoji} {explanation}")

            # Visualization: Expected position changes with cleaner approach
            st.markdown("---")
            st.subheader("📊 Expected Position Changes")

            # Create a simple bar chart showing difference between grid and predicted
            result_sorted = result.sort_values('GridPosition')  # Sort by grid position for better readability

            fig = go.Figure()

            # Calculate color based on position change
            colors = ['green' if row['Ensemble_Pred'] < row['GridPosition']
                     else 'red' if row['Ensemble_Pred'] > row['GridPosition']
                     else 'gray' for _, row in result_sorted.iterrows()]

            # Create bar chart showing the delta
            fig.add_trace(go.Bar(
                x=result_sorted['DriverCode'],
                y=result_sorted['Ensemble_Pred'] - result_sorted['GridPosition'],  # Delta
                marker_color=colors,
                name='Position Change',
                text=[f"{pred:.1f}" if pred < grid else f"-{pred:.1f}"
                      for grid, pred in zip(result_sorted['GridPosition'], result_sorted['Ensemble_Pred'])],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Grid: %{customdata[0]}<br>Predicted: %{customdata[1]:.1f}<br>Change: %{y:.1f}<extra></extra>',
                customdata=result_sorted[['GridPosition', 'Ensemble_Pred']].values
            ))

            fig.update_layout(
                title="Expected Position Changes from Grid Position",
                xaxis_title="Driver",
                yaxis_title="Position Change (Negative = Gains Positions)",
                shapes=[dict(type='line', xref='paper', x0=0, x1=1, yref='y', y0=0, y1=0,
                           line=dict(color='gray', dash='dash', width=1))],  # Zero line
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Additional info for clarity
            st.info("💡 **How to read**: Bars below 0 = drivers gaining positions (prediction better than grid). Bars above 0 = drivers losing positions. Length = magnitude of change.")

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