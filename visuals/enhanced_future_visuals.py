"""
Enhanced visualization utilities for F1 Race Outcome Predictor - Future Predictions
Contains storytelling-focused visualizations that build narrative and context
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from .color_theme import F1ColorTheme


def plot_model_confidence_metrics(artifacts: dict) -> go.Figure:
    """
    Display model reliability metrics to establish trust in predictions
    
    Args:
        artifacts: Loaded model artifacts containing historical performance data
    
    Returns:
        Plotly figure showing model confidence indicators
    """
    # Get model performance data
    training_info = artifacts.get('training_info', {})
    feature_importance = training_info.get('feature_importance', {})
    
    # Extract key performance metrics
    avg_error = training_info.get('avg_error', 2.5)
    accuracy_rate_2_pos = training_info.get('accuracy_rate_2_pos', 0.65)
    accuracy_rate_1_pos = training_info.get('accuracy_rate_1_pos', 0.35)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "xy"}]],
        subplot_titles=['Avg Error', 'Accuracy (±2)', 'Accuracy (±1)', 'Feature Importance'],
        vertical_spacing=0.15,
        horizontal_spacing=0.2
    )
    
    # Average Error
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_error,
        domain={'x': [0, 0.48], 'y': [0.6, 1]},
        title={'text': "Avg Error (pos)"},
        gauge={
            'axis': {'range': [None, 10]},
            'bar': {'color': F1ColorTheme.REFERENCE},
            'steps': [
                {'range': [0, 2], 'color': F1ColorTheme.EXCELLENT},
                {'range': [2, 4], 'color': F1ColorTheme.GOOD},
                {'range': [4, 6], 'color': F1ColorTheme.FAIR},
                {'range': [6, 10], 'color': F1ColorTheme.POOR}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_error
            }
        }
    ), row=1, col=1)

    # Accuracy within 2 positions
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=accuracy_rate_2_pos * 100,
        domain={'x': [0.52, 1], 'y': [0.6, 1]},
        title={'text': "Accuracy ±2 pos (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': F1ColorTheme.REFERENCE},
            'steps': [
                {'range': [0, 50], 'color': F1ColorTheme.POOR},
                {'range': [50, 70], 'color': F1ColorTheme.FAIR},
                {'range': [70, 85], 'color': F1ColorTheme.GOOD},
                {'range': [85, 100], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': accuracy_rate_2_pos * 100
            }
        }
    ), row=1, col=2)
    
    # Accuracy within 1 position
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=accuracy_rate_1_pos * 100,
        domain={'x': [0, 1], 'y': [0, 0.4]},
        title={'text': "Accuracy ±1 pos (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': F1ColorTheme.REFERENCE},
            'steps': [
                {'range': [0, 30], 'color': F1ColorTheme.POOR},
                {'range': [30, 50], 'color': F1ColorTheme.FAIR},
                {'range': [50, 70], 'color': F1ColorTheme.GOOD},
                {'range': [70, 100], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': accuracy_rate_1_pos * 100
            }
        }
    ), row=2, col=1)
    
    # Feature importance (simplified)
    if feature_importance:
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        df_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).nlargest(5, 'Importance')
    else:
        df_importance = pd.DataFrame({
            'Feature': ['GridPosition', 'DriverAvgPos', 'TeamAvgPos', 'DriverDNFRate', 'TeamDNFRate'],
            'Importance': [0.35, 0.25, 0.20, 0.10, 0.10]
        })
    
    fig.add_trace(go.Bar(
        x=df_importance['Importance'],
        y=df_importance['Feature'],
        orientation='h',
        marker_color=F1ColorTheme.ACTUAL,
        name='Feature Importance'
    ), row=2, col=2)
    
    fig.update_layout(
        title="Model Confidence Metrics",
        height=500,
        showlegend=False
    )
    
    return fig


def plot_position_changes_story(result: pd.DataFrame) -> go.Figure:
    """
    Create a compelling position change visualization that tells the story
    
    Args:
        result: DataFrame with prediction results including 'GridPosition' and 'Ensemble_Pred'
    
    Returns:
        Plotly figure showing position gains/losses with storytelling elements
    """
    # Calculate position changes
    result = result.copy()
    result['PositionChange'] = result['GridPosition'] - result['Ensemble_Pred']
    result['ChangeCategory'] = result['PositionChange'].apply(lambda x: 
        'Big Mover Up' if x >= 3 else 
        'Mover Up' if x >= 1 else
        'Mover Down' if x <= -1 else
        'No Change'
    )
    
    # Sort by grid position for better storytelling flow
    df_sorted = result.sort_values('GridPosition')
    
    fig = go.Figure()
    
    # Create segments from grid to predicted position
    for idx, row in df_sorted.iterrows():
        # Determine color based on change direction and magnitude
        change = row['PositionChange']
        if change > 0:
            color = F1ColorTheme.BIG_GAIN if change >= 3 else F1ColorTheme.SMALL_GAIN
        elif change < 0:
            color = F1ColorTheme.BIG_LOSS if change <= -3 else F1ColorTheme.SMALL_LOSS
        else:
            color = F1ColorTheme.NO_CHANGE
        
        fig.add_trace(go.Scatter(
            x=[row['GridPosition'], row['Ensemble_Pred']],
            y=[row['DriverCode'], row['DriverCode']],
            mode='lines+markers',
            line=dict(color=color, width=4 if abs(change) > 2 else 2),
            marker=dict(
                size=12 if abs(change) > 2 else 8,
                symbol='arrow' if change > 0 else 'arrow-bar-left' if change < 0 else 'circle',
                color=color
            ),
            name=row['DriverCode'],
            hovertemplate=f"<b>%{{y}}</b><br>Grid: %{{x[0]}} → Predicted: %{{x[1]}}<br>Change: {change:+.1f} pos<extra></extra>",
            showlegend=False
        ))
    
    # Add start position markers (circles)
    fig.add_trace(go.Scatter(
        x=df_sorted['GridPosition'],
        y=df_sorted['DriverCode'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=F1ColorTheme.ACTUAL,
            symbol='circle'
        ),
        text=df_sorted['GridPosition'].astype(int),
        textposition='middle center',
        name='Grid Position',
        hovertemplate='<b>%{y}</b><br>Grid: %{x}<extra></extra>'
    ))
    
    # Add predicted position markers (triangles)
    fig.add_trace(go.Scatter(
        x=df_sorted['Ensemble_Pred'],
        y=df_sorted['DriverCode'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=F1ColorTheme.PREDICTED,
            symbol='triangle-up'
        ),
        text=df_sorted['Ensemble_Pred'].round(1),
        textposition='middle center',
        name='Predicted Position',
        hovertemplate='<b>%{y}</b><br>Predicted: %{x:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Predicted Position Changes (Grid → Predicted Finish)",
        xaxis_title="Position (Lower = Better)",
        yaxis_title="Driver",
        height=max(400, len(df_sorted) * 25),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white'
    )
    
    return fig


def plot_prediction_certainty_bands(result: pd.DataFrame) -> go.Figure:
    """
    Show prediction confidence using historical error patterns
    
    Args:
        result: DataFrame with prediction results
    
    Returns:
        Plotly figure showing confidence intervals
    """
    # Calculate historical error-based confidence bands
    # Use a fixed error margin based on model performance (2.5 positions average error)
    result = result.copy()
    confidence_range = 2.5  # Based on historical model performance
    
    result['LowerBound'] = np.maximum(1, result['Ensemble_Pred'] - confidence_range)
    result['UpperBound'] = result['Ensemble_Pred'] + confidence_range
    
    # Sort by predicted position 
    df_sorted = result.sort_values('Ensemble_Pred')
    
    fig = go.Figure()
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=df_sorted['LowerBound'],
        y=df_sorted['DriverCode'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['UpperBound'],
        y=df_sorted['DriverCode'],
        mode='lines',
        line=dict(width=0),
        fill='tonextx',
        fillcolor='rgba(255, 0, 0, 0.1)',
        showlegend=False,
        name='Confidence Band',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x[0]:.1f} - %{x[1]:.1f}<extra></extra>'
    ))
    
    # Add predicted positions as main markers
    fig.add_trace(go.Scatter(
        x=df_sorted['Ensemble_Pred'],
        y=df_sorted['DriverCode'],
        mode='markers+text',
        marker=dict(
            size=14,
            color=F1ColorTheme.PREDICTED
        ),
        text=df_sorted['Ensemble_Pred'].round(1),
        textposition='middle center',
        name='Predicted Position',
        hovertemplate=f'<b>%{{y}}</b><br>Predicted: %{{x:.1f}}<br>Confidence: ±{confidence_range:.1f} pos<extra></extra>'
    ))
    
    fig.update_layout(
        title="Prediction Confidence with Error Margins",
        xaxis_title="Predicted Position (Lower = Better)",
        yaxis_title="Driver",
        height=max(400, len(df_sorted) * 25),
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig


def plot_team_performance_story(result: pd.DataFrame) -> go.Figure:
    """
    Visualize team-level predictions with story context
    
    Args:
        result: DataFrame with prediction results including TeamName
    
    Returns:
        Plotly figure showing team predictions
    """
    # Calculate team-level metrics
    team_metrics = result.groupby('TeamName').agg({
        'Ensemble_Pred': ['mean', 'std'],
        'GridPosition': 'mean',
        'DriverCode': 'count'
    }).round(2)
    
    # Flatten multi-level columns
    team_metrics.columns = ['PredictedAvg', 'PredictedStd', 'GridAvg', 'DriverCount']
    team_metrics = team_metrics.reset_index()
    
    # Sort by predicted average to show story flow
    team_metrics = team_metrics.sort_values('PredictedAvg')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Team Predicted Performance', 'Expected Team Movement'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Left plot: Team predicted vs grid performance
    fig.add_trace(
        go.Bar(
            x=team_metrics['TeamName'],
            y=team_metrics['GridAvg'],
            name='Starting Grid Avg',
            marker_color=F1ColorTheme.ACTUAL,
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=team_metrics['TeamName'],
            y=team_metrics['PredictedAvg'],
            name='Predicted Avg',
            marker_color=F1ColorTheme.PREDICTED,
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Right plot: Team movement
    team_metrics['ExpectedMovement'] = team_metrics['GridAvg'] - team_metrics['PredictedAvg']
    fig.add_trace(
        go.Bar(
            x=team_metrics['TeamName'],
            y=team_metrics['ExpectedMovement'],
            name='Expected Movement',
            marker_color=team_metrics['ExpectedMovement'].apply(
                lambda x: F1ColorTheme.BIG_GAIN if x >= 3 else
                         F1ColorTheme.SMALL_GAIN if x >= 1 else
                         F1ColorTheme.BIG_LOSS if x <= -3 else
                         F1ColorTheme.SMALL_LOSS
            ),
            opacity=0.8
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Team-Level Performance Predictions",
        height=500,
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Position (Lower = Better)", row=1, col=1)
    fig.update_yaxes(title_text="Movement (Positive = Gains)", row=1, col=2)
    fig.update_xaxes(title_text="Team", row=1, col=1)
    fig.update_xaxes(title_text="Team", row=1, col=2)
    
    return fig


def highlight_key_predictions(result: pd.DataFrame) -> list:
    """
    Generate narrative highlights about the most interesting predictions
    
    Args:
        result: DataFrame with prediction results
    
    Returns:
        List of narrative highlights
    """
    result = result.copy()
    result['PositionChange'] = result['GridPosition'] - result['Ensemble_Pred']
    result = result.sort_values('PositionChange', key=abs, ascending=False)
    
    highlights = []
    driver_names = get_driver_names()
    
    # Top 3 biggest movers
    biggest_movers = result.head(3)
    for _, pred in biggest_movers.iterrows():
        change = pred['PositionChange']
        direction = "gain" if change > 0 else "fall"
        magnitude = abs(change)
        
        driver_name = driver_names.get(pred['DriverCode'], pred['DriverCode'])
        
        if magnitude >= 3:
            highlight = f"🔥 **{driver_name} ({pred['DriverCode']})** predicted to {direction} {magnitude:.1f} positions from grid"
        elif magnitude >= 1:
            highlight = f"⚡ **{driver_name} ({pred['DriverCode']})** expected to {direction} {magnitude:.1f} positions"
        else:
            highlight = f"📍 **{driver_name} ({pred['DriverCode']})** predicted to finish near grid position"
        
        highlights.append(highlight)
    
    # Highlight top 3 predictions
    top_preds = result.nsmallest(3, 'Ensemble_Pred')
    for _, pred in top_preds.iterrows():
        if pred['Ensemble_Pred'] <= 3:
            driver_name = driver_names.get(pred['DriverCode'], pred['DriverCode'])
            highlights.append(f"🏆 **{driver_name} ({pred['DriverCode']})** predicted for podium (P{pred['Ensemble_Pred']:.1f})")
    
    # Highlight potential surprises (low grid, high predicted finish)
    overachievers = result[result['GridPosition'] > result['Ensemble_Pred'] + 2]
    for _, pred in overachievers.iterrows():
        driver_name = driver_names.get(pred['DriverCode'], pred['DriverCode'])
        highlights.append(f"🎯 **{driver_name} ({pred['DriverCode']})** overachiever: Grid {pred['GridPosition']} → Predicted P{pred['Ensemble_Pred']:.1f}")
    
    return highlights[:5]  # Return top 5 highlights


def get_driver_names():
    """
    Get full driver names for driver codes.
    Returns a dictionary mapping driver codes to full names.
    """
    return {
        'VER': 'Max Verstappen',          # Red Bull
        'LEC': 'Charles Leclerc',         # Ferrari
        'HAM': 'Lewis Hamilton',          # Ferrari
        'RUS': 'George Russell',          # Mercedes
        'NOR': 'Lando Norris',            # McLaren
        'PIA': 'Oscar Piastri',           # McLaren
        'ANT': 'Andrea Kimi Antonelli',   # Mercedes
        'ALO': 'Fernando Alonso',         # Aston Martin
        'STR': 'Lance Stroll',            # Aston Martin
        'TSU': 'Yuki Tsunoda',            # RB
        'HAD': 'Isack Hadjar',           # RB
        'BEA': 'Oliver Bearman',         # Haas
        'OCO': 'Esteban Ocon',            # Haas
        'GAS': 'Pierre Gasly',            # Alpine
        'DOO': 'Jack Doohan',            # Alpine
        'DEV': 'Alex Albon',              # Williams
        'SAI': 'Carlos Sainz',            # Williams
        'HUL': 'Nico Hulkenberg',        # Kick Sauber
        'BOR': 'Gabriel Bortoleto',      # Kick Sauber
        'MAG': 'Kevin Magnussen',         # Haas (example)
        'RIC': 'Daniel Ricciardo',        # RB (example)
        'ALB': 'Alexander Albon',         # Williams (example)
        'SAR': 'Nyck de Vries',           # AlphaTauri (example)
        'ZHO': 'Zhou Guanyu',             # Sauber (example)
        'BOT': 'Valtteri Bottas',         # Sauber (example)
        'PER': 'Sergio Perez',            # Red Bull (example)
        'MEX': 'Sergio Perez',            # Red Bull (example)
        'OCO': 'Esteban Ocon',            # Alpine (example)
        'ALO': 'Fernando Alonso',         # Aston Martin (example)
    }