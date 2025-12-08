#!/usr/bin/env python
"""
Test script to generate visualizations for past race analysis using actual historical data.
This will allow us to evaluate the quality and effectiveness of the visualization system.
"""

import pandas as pd
import numpy as np
from train_model import load_trained_model
from data_loader import F1DataLoader
from feature_builder import FeatureBuilder
from config import FEATURE_COLUMNS, ENSEMBLE_WEIGHTS
from visuals import (
    plot_error_by_driver,
    plot_dumbbell_positions,
    plot_predicted_order,
    plot_gain_loss_vs_grid,
    plot_team_performance_comparison,
    plot_summary_metrics,
    plot_error_distribution
)
from visuals.metrics_calculator import calculate_summary_metrics


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


def test_past_race_analysis():
    """Test the past race analysis visualizations with real data."""
    print("="*60)
    print("TESTING PAST RACE ANALYSIS VISUALIZATIONS")
    print("="*60)
    
    # Load model artifacts
    print("\n1. Loading trained model...")
    artifacts = load_trained_model('f1_model_artifacts.pkl')
    print("✓ Model loaded successfully")
    
    # Load historical race data
    print("\n2. Loading historical race data...")
    loader = F1DataLoader()
    
    # Try to get a specific race to analyze
    # Let's use 2024 Abu Dhabi Grand Prix which should be available
    season = 2024
    event = "Abu Dhabi Grand Prix"
    
    print(f"   Loading {season} {event}...")
    session = loader.load_race_session(season, event)
    
    if session is None:
        print(f"   ❌ Could not load {season} {event}")
        # Try another event
        event = "British Grand Prix"
        print(f"   Trying {season} {event}...")
        session = loader.load_race_session(season, event)
        
        if session is None:
            print(f"   ❌ Could not load {season} {event}")
            # Try 2023 if 2024 doesn't work
            season = 2023
            event = "Abu Dhabi Grand Prix"
            print(f"   Trying {season} {event}...")
            session = loader.load_race_session(season, event)
            if session is None:
                print(f"   ❌ Could not load {season} {event}")
                return
    
    print("✓ Race data loaded successfully")
    
    # Extract results
    race_results = loader.extract_race_results(session)
    if race_results is None:
        print("❌ Could not extract race results")
        return
    
    print(f"✓ {len(race_results)} drivers loaded")
    
    # Add metadata
    circuit_info = loader.get_circuit_info(session)
    race_results['Year'] = season
    race_results['EventName'] = circuit_info['event_name']
    race_results['CircuitName'] = circuit_info['circuit_name']
    race_results['RoundNumber'] = circuit_info['round_number']
    race_results['IsWetRace'] = int(loader.is_wet_race(session))
    
    # Build features (using historical stats from training)
    fb_stats = artifacts['feature_builder']
    temp_fb = FeatureBuilder(pd.DataFrame())  # Empty base data
    temp_fb.driver_stats = fb_stats['driver_stats']
    temp_fb.team_stats = fb_stats['team_stats']
    temp_fb.circuit_stats = fb_stats['circuit_stats']
    temp_fb.driver_map = fb_stats['driver_map']
    temp_fb.team_map = fb_stats['team_map']
    temp_fb.circuit_map = fb_stats['circuit_map']
    
    print("\n3. Building features...")
    features = temp_fb.build_features(race_results)
    
    # Generate predictions
    print("4. Generating predictions...")
    predictions = predict_race(features, artifacts)
    
    # Merge with actual results
    results_df = race_results.copy()
    results_df = results_df.merge(predictions, on=['DriverCode', 'TeamName'], how='left')
    results_df = results_df.dropna(subset=['Position'])
    results_df = results_df.sort_values('Position')
    
    # Calculate metrics
    results_df['AbsError'] = np.abs(results_df['Position'] - results_df['Ensemble_Pred'])
    results_df['Delta'] = results_df['GridPosition'] - results_df['Ensemble_Pred']
    
    print(f"✓ Predictions generated for {len(results_df)} drivers")
    print(f"✓ MAE: {results_df['AbsError'].mean():.2f} positions")
    
    # Test visualizations
    print("\n5. Testing visualizations...")
    
    # Test 1: plot_error_by_driver
    print("   Testing plot_error_by_driver...")
    try:
        fig1 = plot_error_by_driver(results_df)
        print("   ✓ plot_error_by_driver works correctly")
    except Exception as e:
        print(f"   ✗ plot_error_by_driver failed: {e}")
    
    # Test 2: plot_dumbbell_positions
    print("   Testing plot_dumbbell_positions...")
    try:
        fig2 = plot_dumbbell_positions(results_df)
        print("   ✓ plot_dumbbell_positions works correctly")
    except Exception as e:
        print(f"   ✗ plot_dumbbell_positions failed: {e}")
    
    # Test 3: plot_gain_loss_vs_grid
    print("   Testing plot_gain_loss_vs_grid...")
    try:
        fig3 = plot_gain_loss_vs_grid(results_df[['DriverCode', 'Delta', 'GridPosition', 'Ensemble_Pred', 'TeamName']])
        print("   ✓ plot_gain_loss_vs_grid works correctly")
    except Exception as e:
        print(f"   ✗ plot_gain_loss_vs_grid failed: {e}")
    
    # Test 4: plot_team_performance_comparison
    print("   Testing plot_team_performance_comparison...")
    try:
        fig4 = plot_team_performance_comparison(results_df)
        print("   ✓ plot_team_performance_comparison works correctly")
    except Exception as e:
        print(f"   ✗ plot_team_performance_comparison failed: {e}")
    
    # Test 5: plot_summary_metrics
    print("   Testing plot_summary_metrics...")
    try:
        metrics = calculate_summary_metrics(results_df)
        fig5 = plot_summary_metrics(metrics)
        print("   ✓ plot_summary_metrics works correctly")
    except Exception as e:
        print(f"   ✗ plot_summary_metrics failed: {e}")
    
    # Test 6: plot_error_distribution
    print("   Testing plot_error_distribution...")
    try:
        fig6 = plot_error_distribution(results_df)
        print("   ✓ plot_error_distribution works correctly")
    except Exception as e:
        print(f"   ✗ plot_error_distribution failed: {e}")
    
    # Display results summary
    print("\n6. RESULTS SUMMARY")
    print(f"   - Race: {season} {event}")
    print(f"   - Drivers: {len(results_df)}")
    print(f"   - MAE: {results_df['AbsError'].mean():.2f} positions")
    print(f"   - Median Error: {results_df['AbsError'].median():.2f} positions")
    print(f"   - Accurate predictions (<2 pos): {(results_df['AbsError'] < 2).sum()}/{len(results_df)} ({(results_df['AbsError'] < 2).mean()*100:.1f}%)")
    
    print(f"\n   Top 5 most accurate predictions:")
    top_5 = results_df.nsmallest(5, 'AbsError')
    for _, row in top_5.iterrows():
        print(f"     {row['DriverCode']}: P{int(row['Position'])} actual vs P{row['Ensemble_Pred']:.1f} predicted (error: {row['AbsError']:.2f})")
    
    print(f"\n   Top 5 least accurate predictions:")
    bottom_5 = results_df.nlargest(5, 'AbsError')
    for _, row in bottom_5.iterrows():
        print(f"     {row['DriverCode']}: P{int(row['Position'])} actual vs P{row['Ensemble_Pred']:.1f} predicted (error: {row['AbsError']:.2f})")
    
    print("\n7. ANALYSIS COMPLETE")
    
    return results_df


def generate_visualization_roast():
    """Generate a detailed analysis of the visualization system."""
    print("\n" + "="*80)
    print("ROASTING THE F1 RACE VISUALIZATION SYSTEM")
    print("="*80)
    
    print("\n🎯 VISUALIZATION STRENGTHS:")
    print("  • Consistent color scheme (F1ColorTheme) - helps with readability")
    print("  • Interactive Plotly visualizations with hover details")
    print("  • Multiple visualization types (error, scatter, dumbbell, etc.)")
    print("  • Contextual explanations in figure titles and annotations")
    print("  • Error-based color coding for immediate visual assessment")
    
    print("\n⚠️  VISUALIZATION WEAKNESSES:")
    print("  • plot_error_by_driver() has issues with missing 'TeamName' column in test")
    print("  • plot_predicted_order() has issues with missing 'GridPosition' column in test")
    print("  • Some visualizations don't handle edge cases well (empty data, NaN values)")
    print("  • Dumbbell plot can be cluttered with many drivers (20 drivers may be too much)")
    print("  • Color schemes may not be colorblind-friendly")
    
    print("\n📊 DETAILED CRITIQUE:")
    print("  • plot_error_by_driver: Good concept but needs better error handling")
    print("  • plot_dumbbell_positions: Visually appealing but can be cluttered")
    print("  • plot_predicted_order: Confusing when used for historical analysis")
    print("  • plot_gain_loss_vs_grid: Good for showing movement, clear direction indicators")
    print("  • plot_team_performance_comparison: Useful aggregation but may oversimplify")
    
    print("\n🔧 POTENTIAL IMPROVEMENTS:")
    print("  • Add validation to ensure required columns exist before plotting")
    print("  • Implement better handling for different sample sizes")
    print("  • Add colorblind-friendly color palettes")
    print("  • Add option for horizontal vs vertical layout based on driver count")
    print("  • Add better legends and annotations")
    print("  • More responsive sizing based on data volume")
    
    print("\n🎯 OVERALL ASSESSMENT:")
    print("  • The visualization system shows good intent with comprehensive analysis")
    print("  • Color coding and interactive elements enhance usability")
    print("  • However, implementation has some bugs and robustness issues")
    print("  • The visualizations provide good insights into model performance")
    print("  • Could benefit from more rigorous error handling and validation")


if __name__ == "__main__":
    print("Testing past race visualization system...")
    results_df = test_past_race_analysis()
    generate_visualization_roast()