#!/usr/bin/env python
"""
Test the comprehensive past race analysis functionality.
"""

import pandas as pd
import numpy as np
from train_model import load_trained_model
from data_loader import F1DataLoader
from feature_builder import FeatureBuilder
from config import FEATURE_COLUMNS, ENSEMBLE_WEIGHTS
from visuals import (
    plot_comprehensive_performance_dashboard, 
    plot_position_accuracy_heatmap,
    add_prediction_methodology_explanation
)


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


def test_comprehensive_analysis():
    """Test the comprehensive past race analysis."""
    print("Testing comprehensive past race analysis...")
    
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
    print(f"✓ Model MAE: {results_df['AbsError'].mean():.2f} positions")
    
    # Test the comprehensive dashboard
    try:
        dashboard_fig = plot_comprehensive_performance_dashboard(results_df)
        print("✓ Comprehensive performance dashboard created successfully!")
        
        # Calculate some metrics to verify
        mae = results_df['AbsError'].mean()
        perfect_preds = (results_df['AbsError'] < 0.5).sum()
        within_2_pos = (results_df['AbsError'] < 2).sum()
        print(f"  - Model MAE: {mae:.2f}")
        print(f"  - Perfect predictions: {perfect_preds}/{len(results_df)} ({perfect_preds/len(results_df)*100:.1f}%)")
        print(f"  - Within ±2 positions: {within_2_pos}/{len(results_df)} ({within_2_pos/len(results_df)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test the heatmap
    try:
        heatmap_fig = plot_position_accuracy_heatmap(results_df)
        print("✓ Position accuracy heatmap created successfully!")
        
        # Show accuracy by position range
        pos_ranges = pd.cut(results_df['Position'], bins=range(0, 22, 4), include_lowest=True, labels=['P1-4', 'P5-8', 'P9-12', 'P13-16', 'P17-20'])
        for pos_range in pos_ranges.cat.categories:
            range_mask = pos_ranges == pos_range
            if range_mask.any():
                range_error = results_df[range_mask]['AbsError'].mean()
                print(f"  - {pos_range} accuracy: {range_error:.2f} positions off")
        
    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test methodology explanation
    try:
        explanation = add_prediction_methodology_explanation()
        print("✓ Methodology explanation created successfully!")
        print("  - Explanation length:", len(explanation), "characters")
    except Exception as e:
        print(f"❌ Error creating explanation: {e}")
        return False
    
    # Test baseline comparison
    grid_errors = np.abs(results_df['GridPosition'] - results_df['Position'])
    grid_mae = grid_errors.mean()
    model_improvement = grid_mae - results_df['AbsError'].mean()
    
    print(f"\n📊 Baseline Comparison:")
    print(f"  - Grid position MAE: {grid_mae:.2f}")
    print(f"  - Model MAE: {results_df['AbsError'].mean():.2f}")
    print(f"  - Improvement: {model_improvement:.2f} positions better than grid")
    
    return True


if __name__ == "__main__":
    print("Testing comprehensive past race analysis...")
    print("=" * 60)
    
    success = test_comprehensive_analysis()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ COMPREHENSIVE ANALYSIS SUCCESSFULLY IMPLEMENTED!")
        print("✅ Multiple metrics now available for past race validation")
        print("✅ Baseline comparisons included")
        print("✅ Position-based accuracy analysis added")
        print("✅ Clear methodology explanation provided")
    else:
        print("\n❌ Test failed")