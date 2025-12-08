#!/usr/bin/env python
"""
Test script to verify that visualization functions work correctly.
"""

import pandas as pd
import numpy as np
from visuals import (
    plot_error_by_driver,
    plot_dumbbell_positions,
    plot_predicted_order,
    plot_gain_loss_vs_grid
)

def test_visualizations():
    """Test that all visualization functions work with sample data"""
    print("Testing visualization functions...")
    
    # Test 1: plot_error_by_driver
    print("\n1. Testing plot_error_by_driver...")
    try:
        df1 = pd.DataFrame({
            'DriverCode': ['VER', 'HAM', 'LEC', 'MAG', 'RIC'],
            'AbsError': [0.5, 2.3, 1.1, 3.2, 1.8],
            'Position': [1, 2, 3, 15, 10],
            'Ensemble_Pred': [1.2, 2.8, 2.9, 12, 9.5]
        })
        fig1 = plot_error_by_driver(df1)
        print("   ✓ plot_error_by_driver works correctly")
    except Exception as e:
        print(f"   ✗ plot_error_by_driver failed: {e}")
    
    # Test 2: plot_dumbbell_positions
    print("\n2. Testing plot_dumbbell_positions...")
    try:
        df2 = pd.DataFrame({
            'DriverCode': ['VER', 'HAM', 'LEC', 'MAG', 'RIC'],
            'Position': [1, 2, 3, 15, 10],
            'Ensemble_Pred': [1.2, 3.5, 2.9, 12, 8.5],
            'TeamName': ['Red Bull', 'Ferrari', 'Ferrari', 'Haas', 'AlphaTauri']
        })
        fig2 = plot_dumbbell_positions(df2)
        print("   ✓ plot_dumbbell_positions works correctly")
    except Exception as e:
        print(f"   ✗ plot_dumbbell_positions failed: {e}")
    
    # Test 3: plot_predicted_order
    print("\n3. Testing plot_predicted_order...")
    try:
        df3 = pd.DataFrame({
            'DriverCode': ['VER', 'HAM', 'LEC', 'MAG', 'RIC'],
            'Ensemble_Pred': [1.2, 2.8, 2.9, 12.5, 8.5],
            'TeamName': ['Red Bull', 'Ferrari', 'Ferrari', 'Haas', 'AlphaTauri']
        })
        fig3 = plot_predicted_order(df3)
        print("   ✓ plot_predicted_order works correctly")
    except Exception as e:
        print(f"   ✗ plot_predicted_order failed: {e}")
    
    # Test 4: plot_gain_loss_vs_grid
    print("\n4. Testing plot_gain_loss_vs_grid...")
    try:
        df4 = pd.DataFrame({
            'DriverCode': ['VER', 'HAM', 'LEC', 'MAG', 'RIC'],
            'Delta': [0.8, -1.5, 0.2, 2.5, -0.5],
            'GridPosition': [1, 3, 2, 18, 9],
            'Ensemble_Pred': [1.2, 4.5, 2.2, 15.5, 9.5],
            'TeamName': ['Red Bull', 'Ferrari', 'Ferrari', 'Haas', 'AlphaTauri']
        })
        fig4 = plot_gain_loss_vs_grid(df4)
        print("   ✓ plot_gain_loss_vs_grid works correctly")
    except Exception as e:
        print(f"   ✗ plot_gain_loss_vs_grid failed: {e}")
    
    print("\nAll visualization tests completed!")

if __name__ == "__main__":
    test_visualizations()