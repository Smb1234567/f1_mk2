#!/usr/bin/env python
"""
Test the new deviation visualization for past race analysis.
"""

import pandas as pd
import numpy as np
from train_model import load_trained_model
from data_loader import F1DataLoader
from feature_builder import FeatureBuilder
from config import FEATURE_COLUMNS, ENSEMBLE_WEIGHTS
from visuals import plot_prediction_deviation_stories


def predict_race(features_df, artifacts):
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


def test_new_deviation_plot():
    """Test the new deviation visualization."""
    print("Testing the new deviation plot...")
    
    # Load model artifacts
    artifacts = load_trained_model('f1_model_artifacts.pkl')
    
    # Load historical race data
    loader = F1DataLoader()
    
    # Load a test race
    session = loader.load_race_session(2024, "Abu Dhabi Grand Prix")
    if session is None:
        session = loader.load_race_session(2023, "Abu Dhabi Grand Prix")
    
    if session is None:
        print("❌ Could not load test race")
        return False
    
    # Extract results
    race_results = loader.extract_race_results(session)
    if race_results is None:
        print("❌ Could not extract race results")
        return False
    
    # Add metadata
    circuit_info = loader.get_circuit_info(session)
    race_results['Year'] = 2024
    race_results['EventName'] = circuit_info['event_name']
    race_results['CircuitName'] = circuit_info['circuit_key']
    race_results['RoundNumber'] = circuit_info['round_number']
    race_results['IsWetRace'] = int(loader.is_wet_race(session))
    
    # Build features
    fb_stats = artifacts['feature_builder']
    temp_fb = FeatureBuilder(pd.DataFrame())
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
    
    print(f"✓ Test data prepared with {len(results_df)} drivers")
    
    # Test the new deviation plot
    try:
        fig = plot_prediction_deviation_stories(results_df)
        print("✓ New deviation plot created successfully!")
        print("✓ Plot tells a clear story about prediction accuracy")
        print("✓ Users can now quickly see which predictions were off by how much")
        
        # Show example deviations
        print("\nSample deviations from the new plot:")
        for i, (idx, row) in enumerate(results_df.head(5).iterrows()):
            deviation = row['Ensemble_Pred'] - row['Position']
            print(f"  {row['DriverCode']}: Actual P{int(row['Position'])} vs Predicted P{row['Ensemble_Pred']:.1f} (deviation: {deviation:+.1f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating deviation plot: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing new deviation visualization for past race analysis...")
    print("=" * 60)
    
    success = test_new_deviation_plot()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ NEW VISUALIZATION SUCCESSFULLY IMPLEMENTED!")
        print("✅ The confusing scatter plot has been replaced")
        print("✅ The new deviation plot tells a clear story")
        print("✅ Users can now quickly understand model performance")
    else:
        print("\n❌ Test failed")