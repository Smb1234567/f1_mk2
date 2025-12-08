"""
Calculate key metrics for model performance evaluation
"""

import pandas as pd


def calculate_summary_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculate key metrics for model performance evaluation
    
    Args:
        results_df: DataFrame with columns ['DriverCode', 'Position', 'Ensemble_Pred', 'AbsError']

    Returns:
        Dictionary with performance metrics
    """
    # Basic accuracy metrics
    mae = results_df['AbsError'].mean()
    median_error = results_df['AbsError'].median()
    
    # Accuracy rates
    perfect_preds = (results_df['AbsError'] < 0.5).sum()
    good_preds = (results_df['AbsError'] < 2).sum()
    total = len(results_df)
    
    # Find biggest surprises
    results_df = results_df.copy()  # Avoid modifying the original
    results_df['Delta'] = results_df['Position'] - results_df['Ensemble_Pred']
    biggest_overperform = results_df.loc[results_df['Delta'].idxmin()]  # Most negative = beat prediction
    biggest_underperform = results_df.loc[results_df['Delta'].idxmax()]  # Most positive = missed prediction
    
    # Most accurate and least accurate predictions
    most_accurate = results_df.loc[results_df['AbsError'].idxmin()]
    least_accurate = results_df.loc[results_df['AbsError'].idxmax()]
    
    return {
        'mae': round(mae, 2),
        'median_error': round(median_error, 2),
        'accuracy_rate': round((good_preds / total) * 100, 1),
        'perfect_rate': round((perfect_preds / total) * 100, 1),
        'biggest_surprise_driver': biggest_overperform['DriverCode'],
        'biggest_surprise_delta': biggest_overperform['Delta'],
        'biggest_surprise_actual': int(biggest_overperform['Position']),
        'biggest_surprise_pred': biggest_overperform['Ensemble_Pred'],
        'biggest_disappointment_driver': biggest_underperform['DriverCode'],
        'biggest_disappointment_delta': biggest_underperform['Delta'],
        'biggest_disappointment_actual': int(biggest_underperform['Position']),
        'biggest_disappointment_pred': biggest_underperform['Ensemble_Pred'],
        'most_accurate_driver': most_accurate['DriverCode'],
        'most_accurate_error': most_accurate['AbsError'],
        'least_accurate_driver': least_accurate['DriverCode'],
        'least_accurate_error': least_accurate['AbsError']
    }