#!/usr/bin/env python
"""
Test script to verify that the explanation function provides better narrative content.
"""

import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS

def mock_artifacts():
    """Mock artifacts to test the explanation function without loading the full model"""
    return {
        'training_info': {'shap_available': False},  # Test fallback path
        'models': {},
        'imputer': None,
        'shap_explainer': None
    }

def test_explanation():
    """Test the explanation function"""
    print("Testing explanation function...")
    
    # Import the function
    import sys
    sys.path.append('/home/igris94/f1_predictor')
    from app import explain_prediction_with_shap
    
    # Create a mock features row
    features_row = pd.Series({
        'GridPosition': 1,
        'DriverAvgPos': 2.5,
        'TeamAvgPos': 3.2,
        'DriverDNFRate': 0.05,
        'TeamDNFRate': 0.08,
        'GridAdvantage': -1.5,
        'TeamReliability': 0.92,
        'IsWetRace': 0,
        **{col: 0 for col in FEATURE_COLUMNS if col not in [
            'GridPosition', 'DriverAvgPos', 'TeamAvgPos', 
            'DriverDNFRate', 'TeamDNFRate', 'GridAdvantage', 
            'TeamReliability', 'IsWetRace'
        ]}
    })
    
    # Test the explanation function
    artifacts = mock_artifacts()
    explanation = explain_prediction_with_shap('VER', 'Red Bull', features_row, artifacts, 1.8)
    
    print(f"\nExplanation: {explanation}")
    
    # Check if the explanation contains narrative elements
    if 'predicted to finish P' in explanation and ('Red Bull' in explanation or 'driver' in explanation.lower()):
        print("✓ Explanation function works and contains narrative elements")
    else:
        print("✗ Explanation function may not be working as expected")

if __name__ == "__main__":
    test_explanation()