"""
Visualization utilities for F1 Race Outcome Predictor
Contains all plotting functions with consistent styling
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from .color_theme import F1ColorTheme
from .metrics_calculator import calculate_summary_metrics


def plot_summary_metrics(metrics: dict) -> go.Figure:
    """
    Create summary metrics visualization
    
    Args:
        metrics: Dictionary from calculate_summary_metrics()

    Returns:
        Plotly figure with indicator gauges
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "domain", "colspan": 2}, None]],
        vertical_spacing=0.3
    )
    
    # Top Left: Accuracy Rate
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=metrics['accuracy_rate'],
        title={
            "text": "Model Accuracy<br><span style='font-size:0.7em; color:gray'>% Within 2 Positions</span>",
            "font": {"size": 20}
        },
        delta={'reference': 70, 'relative': False},
        number={'suffix': "%", 'font': {'size': 50}},
        domain={'x': [0, 0.5], 'y': [0.5, 1]}
    ), row=1, col=1)
    
    # Top Right: Average Error
    error_color = F1ColorTheme.get_error_color(metrics['mae'])
    fig.add_trace(go.Indicator(
        mode="number",
        value=metrics['mae'],
        title={
            "text": "Avg Prediction Error<br><span style='font-size:0.7em; color:gray'>Positions Off</span>",
            "font": {"size": 20}
        },
        number={'font': {'size': 50, 'color': error_color}},
        domain={'x': [0.5, 1], 'y': [0.5, 1]}
    ), row=1, col=2)
    
    fig.update_layout(
        title={
            'text': "Race Prediction Summary",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2D3436'}
        },
        height=500,
        plot_bgcolor='white',
        paper_bgcolor=F1ColorTheme.BACKGROUND,
        annotations=[
            # Biggest Surprise
            dict(
                x=0.25, y=0.25,
                xref='paper', yref='paper',
                text=f"<b>Biggest Surprise</b><br>" +
                     f"<b>{metrics['biggest_surprise_driver']}</b> finished P{metrics['biggest_surprise_actual']}<br>" +
                     f"(Predicted P{metrics['biggest_surprise_pred']:.1f})",
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(0, 184, 148, 0.15)',
                bordercolor='#00B894',
                borderwidth=2,
                align='center'
            ),
            # Biggest Disappointment
            dict(
                x=0.75, y=0.25,
                xref='paper', yref='paper',
                text=f"<b>Biggest Disappointment</b><br>" +
                     f"<b>{metrics['biggest_disappointment_driver']}</b> finished P{metrics['biggest_disappointment_actual']}<br>" +
                     f"(Predicted P{metrics['biggest_disappointment_pred']:.1f})",
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(225, 112, 85, 0.15)',
                bordercolor='#E17055',
                borderwidth=2,
                align='center'
            )
        ]
    )
    
    return fig


def plot_error_distribution(results_df: pd.DataFrame) -> go.Figure:
    """
    Show how prediction errors are distributed
    
    Args:
        results_df: DataFrame with 'AbsError' column

    Returns:
        Plotly figure showing error distribution histogram
    """
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=results_df['AbsError'],
        nbinsx=15,
        marker_color=F1ColorTheme.ACTUAL,
        opacity=0.7,
        name='Prediction Errors',
        hovertemplate='Error: %{x:.1f} positions<br>Drivers: %{y}<extra></extra>'
    ))
    
    # Add average line
    mae = results_df['AbsError'].mean()
    fig.add_vline(
        x=mae,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Average: {mae:.2f}",
        annotation_position="top right",
        annotation_font_size=14
    )
    
    # Add performance zones
    fig.add_vrect(
        x0=0, x1=2,
        fillcolor=F1ColorTheme.EXCELLENT,
        opacity=0.15,
        line_width=0,
        annotation_text="Excellent<br>(0-2 pos)",
        annotation_position="top left",
        annotation_font_size=11
    )
    
    fig.add_vrect(
        x0=2, x1=5,
        fillcolor=F1ColorTheme.FAIR,
        opacity=0.15,
        line_width=0,
        annotation_text="Fair<br>(2-5 pos)",
        annotation_position="top",
        annotation_font_size=11
    )
    
    fig.add_vrect(
        x0=5, x1=results_df['AbsError'].max() + 0.5,
        fillcolor=F1ColorTheme.POOR,
        opacity=0.15,
        line_width=0,
        annotation_text="Poor<br>(5+ pos)",
        annotation_position="top right",
        annotation_font_size=11
    )
    
    fig.update_layout(
        title={
            'text': "Prediction Error Distribution",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Prediction Error (Positions Off)",
        yaxis_title="Number of Drivers",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='lightgray',
            showgrid=True
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )
    
    return fig


def plot_dumbbell_positions(results_df: pd.DataFrame) -> go.Figure:
    """
    Dumbbell plot showing actual vs predicted positions
    
    Args:
        results_df: DataFrame with columns ['DriverCode', 'Position', 'Ensemble_Pred', 'TeamName']

    Returns:
        Plotly figure for dumbbell plot
    """
    # Sort by actual finishing position for clarity
    df_sorted = results_df.sort_values('Position', ascending=True).copy()
    
    fig = go.Figure()
    
    # Add connecting lines showing gap between actual and predicted
    for idx, row in df_sorted.iterrows():
        # Calculate delta: negative means beat prediction, positive means missed
        delta = row['Position'] - row['Ensemble_Pred']
        color = F1ColorTheme.get_delta_color(-delta)  # Invert for color (beat = green)
        
        fig.add_trace(go.Scatter(
            x=[row['Position'], row['Ensemble_Pred']],
            y=[row['DriverCode'], row['DriverCode']],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add actual position markers (circles)
    fig.add_trace(go.Scatter(
        x=df_sorted['Position'],
        y=df_sorted['DriverCode'],
        mode='markers+text',
        marker=dict(
            size=14,
            color=F1ColorTheme.ACTUAL,
            symbol='circle',
            line=dict(width=2, color='white')
        ),
        text=df_sorted['Position'].astype(int),
        textposition='middle center',
        textfont=dict(color='white', size=10, family='Arial Black'),
        name='Actual Finish',
        hovertemplate='<b>%{y}</b><br>' +
                      'Actual: P%{x}<br>' +
                      'Predicted: P%{customdata:.1f}<extra></extra>',
        customdata=df_sorted['Ensemble_Pred']
    ))
    
    # Add predicted position markers (triangles)
    fig.add_trace(go.Scatter(
        x=df_sorted['Ensemble_Pred'],
        y=df_sorted['DriverCode'],
        mode='markers+text',
        marker=dict(
            size=14,
            color=F1ColorTheme.PREDICTED,
            symbol='triangle-up',
            line=dict(width=2, color='white')
        ),
        text=df_sorted['Ensemble_Pred'].round(1),
        textposition='middle center',
        textfont=dict(color='white', size=9, family='Arial Black'),
        name='Predicted Finish',
        hovertemplate='<b>%{y}</b><br>' +
                      'Predicted: P%{x:.1f}<br>' +
                      'Actual: P%{customdata}<extra></extra>',
        customdata=df_sorted['Position']
    ))
    
    fig.update_layout(
        title={
            'text': "Actual vs Predicted Positions",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Position (Lower = Better)",
        yaxis_title="Driver",
        xaxis=dict(
            autorange=False,
            range=[0.5, df_sorted[['Position', 'Ensemble_Pred']].max().max() + 0.5],
            dtick=1,
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5
        ),
        yaxis=dict(
            gridcolor='lightgray',
            showgrid=False
        ),
        height=max(450, len(df_sorted) * 28),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=80, r=20, t=100, b=60)
    )
    
    return fig


def plot_error_by_driver(results_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing prediction accuracy by driver
    
    Args:
        results_df: DataFrame with columns ['DriverCode', 'AbsError', 'Position', 'Ensemble_Pred', 'TeamName']

    Returns:
        Plotly figure object
    """
    # Sort by error descending to highlight problem predictions
    df_sorted = results_df.sort_values('AbsError', ascending=False).copy()
    
    # Apply consistent color theme
    colors = [F1ColorTheme.get_error_color(err) for err in df_sorted['AbsError']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['DriverCode'],
        x=df_sorted['AbsError'],
        orientation='h',
        marker_color=colors,
        text=df_sorted['AbsError'].round(1),
        textposition='auto',
        textfont=dict(color='white', size=11, family='Arial Black'),
        name='Prediction Error',
        customdata=df_sorted[['Position', 'Ensemble_Pred', 'TeamName']].values,
        hovertemplate='<b>%{y}</b> (%{customdata[2]})<br>' +
                      'Actual: P%{customdata[0]}<br>' +
                      'Predicted: P%{customdata[1]:.1f}<br>' +
                      'Error: %{x:.1f} positions<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Prediction Accuracy by Driver",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Prediction Error (Positions)",
        yaxis_title="Driver",
        height=max(420, len(df_sorted) * 26),
        yaxis=dict(
            autorange='reversed',
            tickfont={'size': 12},
            gridcolor='lightgray',
            showgrid=False
        ),
        xaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5
        ),
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=80, r=20, t=70, b=60)
    )
    
    return fig


def plot_actual_vs_predicted_scatter(results_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot showing actual vs predicted positions
    
    Args:
        results_df: DataFrame with columns ['Position', 'Ensemble_Pred', 'DriverCode', 'TeamName']

    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Calculate max for perfect prediction line
    max_pos = max(results_df['Position'].max(), results_df['Ensemble_Pred'].max()) + 1
    
    # Add perfect prediction reference line (y=x)
    fig.add_trace(go.Scatter(
        x=[1, max_pos],
        y=[1, max_pos],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Color code by error magnitude
    colors = [F1ColorTheme.get_error_color(err) for err in results_df['AbsError']]
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=results_df['Position'],
        y=results_df['Ensemble_Pred'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=results_df['DriverCode'],
        textposition='top center',
        textfont=dict(size=9),
        customdata=results_df[['Position', 'Ensemble_Pred', 'TeamName', 'AbsError']].values,
        hovertemplate='<b>%{text}</b> (%{customdata[2]})<br>' +
                      'Actual: P%{customdata[0]}<br>' +
                      'Predicted: P%{customdata[1]:.1f}<br>' +
                      'Error: %{customdata[3]:.1f} positions<extra></extra>',
        name='Drivers'
    ))
    
    fig.update_layout(
        title={
            'text': "Model Performance: Actual vs Predicted",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Actual Position",
        yaxis_title="Predicted Position",
        xaxis=dict(
            dtick=2,
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5
        ),
        yaxis=dict(
            dtick=2,
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5
        ),
        plot_bgcolor='white',
        width=700,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=11),
        margin=dict(l=60, r=20, t=80, b=60)
    )
    
    return fig


def plot_predicted_order(result: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing predicted finishing order
    
    Args:
        result: DataFrame with columns ['DriverCode', 'Ensemble_Pred', 'TeamName']

    Returns:
        Plotly figure object
    """
    df_sorted = result.sort_values('Ensemble_Pred', ascending=True).copy()
    
    # Team color mapping with better palette
    unique_teams = df_sorted['TeamName'].unique()
    colors = px.colors.qualitative.Set2[:len(unique_teams)]
    team_color_map = dict(zip(unique_teams, colors))
    
    fig = go.Figure()
    
    # Add trace for each team
    for team in unique_teams:
        team_data = df_sorted[df_sorted['TeamName'] == team]
        if not team_data.empty:
            fig.add_trace(go.Bar(
                y=team_data['DriverCode'],
                x=team_data['Ensemble_Pred'],
                orientation='h',
                marker_color=team_color_map[team],
                text=team_data['Ensemble_Pred'].round(1),
                textposition='auto',
                textfont=dict(color='black', size=11, family='Arial Black'),
                name=team,
                customdata=team_data[['TeamName', 'GridPosition']].values,
                hovertemplate='<b>%{y}</b><br>' +
                              'Team: %{customdata[0]}<br>' +
                              'Predicted: P%{x:.1f}<br>' +
                              'Grid: P%{customdata[1]}<extra></extra>',
                showlegend=True
            ))
    
    fig.update_layout(
        title={
            'text': "Predicted Race Finish Order",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Predicted Position (Lower = Better)",
        yaxis_title="Driver",
        height=max(450, len(df_sorted) * 28),
        yaxis=dict(
            autorange='reversed',
            tickfont={'size': 12},
            gridcolor='lightgray',
            showgrid=False
        ),
        xaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5
        ),
        showlegend=True,
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=80, r=160, t=70, b=60),
        legend=dict(
            title="Team",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )
    
    return fig


def plot_gain_loss_vs_grid(result: pd.DataFrame) -> go.Figure:
    """
    Diverging bar chart showing expected position changes
    Delta = GridPosition - Predicted (positive = gain, negative = loss)
    
    Args:
        result: DataFrame with columns ['DriverCode', 'Delta', 'GridPosition', 'Ensemble_Pred', 'TeamName']
    
    Returns:
        Plotly figure object
    """
    # Sort by absolute delta to highlight biggest movers
    df_sorted = result.sort_values('Delta', key=abs, ascending=False).copy()
    
    # Apply consistent color theme
    colors = [F1ColorTheme.get_delta_color(delta) for delta in df_sorted['Delta']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted['DriverCode'],
        y=df_sorted['Delta'],
        marker_color=colors,
        name='Position Change',
        text=[f"+{d:.1f}" if d > 0 else f"{d:.1f}" for d in df_sorted['Delta']],
        textposition='outside',
        textfont=dict(color='black', size=11, family='Arial Black'),
        customdata=df_sorted[['GridPosition', 'Ensemble_Pred', 'TeamName']].values,
        hovertemplate='<b>%{x}</b> (%{customdata[2]})<br>' +
                      'Starting Grid: P%{customdata[0]}<br>' +
                      'Predicted Finish: P%{customdata[1]:.1f}<br>' +
                      'Expected Change: %{y:+.1f} positions<extra></extra>'
    ))
    
    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        line_width=1.5
    )
    
    fig.update_layout(
        title={
            'text': "Expected Position Changes vs Grid",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Driver",
        yaxis_title="Position Change (+ = Gain, - = Loss)",
        height=max(450, len(df_sorted) * 22),
        yaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        xaxis=dict(
            gridcolor='lightgray',
            showgrid=False
        ),
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=60, r=20, t=70, b=100)
    )
    
    return fig


def plot_team_performance_comparison(results_df: pd.DataFrame) -> go.Figure:
    """
    Team-level analysis showing actual vs predicted performance
    
    Args:
        results_df: DataFrame with columns ['DriverCode', 'Position', 'Ensemble_Pred', 'TeamName']
    
    Returns:
        Plotly figure with grouped bars and error line
    """
    # Calculate team statistics
    team_summary = results_df.groupby('TeamName').agg({
        'Position': 'mean',
        'Ensemble_Pred': 'mean',
        'AbsError': 'mean'
    }).round(2).reset_index()
    
    # Sort by actual performance
    team_summary = team_summary.sort_values('Position')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Actual team performance
    fig.add_trace(
        go.Bar(
            x=team_summary['TeamName'],
            y=team_summary['Position'],
            name='Actual Avg Position',
            marker_color=F1ColorTheme.ACTUAL,
            opacity=0.8,
            text=team_summary['Position'].round(1),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Actual: P%{y:.1f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Predicted team performance
    fig.add_trace(
        go.Bar(
            x=team_summary['TeamName'],
            y=team_summary['Ensemble_Pred'],
            name='Predicted Avg Position',
            marker_color=F1ColorTheme.PREDICTED,
            opacity=0.8,
            text=team_summary['Ensemble_Pred'].round(1),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Predicted: P%{y:.1f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Prediction error line
    fig.add_trace(
        go.Scatter(
            x=team_summary['TeamName'],
            y=team_summary['AbsError'],
            mode='lines+markers',
            name='Avg Error',
            line=dict(color=F1ColorTheme.REFERENCE, width=3),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>Avg Error: %{y:.1f} pos<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title={
            'text': "Team Performance: Actual vs Predicted Averages",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Team",
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=60, r=60, t=70, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group'
    )
    
    fig.update_yaxes(
        title_text="Average Position (Lower = Better)",
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Average Error",
        secondary_y=True,
        tickfont_color=F1ColorTheme.REFERENCE
    )

    return fig


def plot_prediction_deviation_stories(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a clear deviation plot showing prediction accuracy with storytelling elements.
    Shows how many positions off each prediction was from actual results.

    Args:
        results_df: DataFrame with columns ['DriverCode', 'Position', 'Ensemble_Pred', 'AbsError', 'TeamName']

    Returns:
        Plotly figure showing prediction deviations with clear narrative
    """
    # Calculate deviation (positive = predicted worse than actual, negative = predicted better than actual)
    df_plot = results_df.copy()
    df_plot['Deviation'] = df_plot['Ensemble_Pred'] - df_plot['Position']  # Positive = overestimated finish position (predicted worse)
    df_plot['Accuracy_Category'] = df_plot['AbsError'].apply(
        lambda x: 'Excellent (0-1)' if x <= 1 else
                 'Good (1-2)' if x <= 2 else
                 'Fair (2-4)' if x <= 4 else
                 'Poor (4+)'
    )

    # Sort by absolute deviation to show biggest errors first
    df_plot = df_plot.sort_values('AbsError', key=abs, ascending=False)

    fig = go.Figure()

    # Add bars for each driver showing deviation
    colors = [F1ColorTheme.get_error_color(err) for err in df_plot['AbsError']]

    fig.add_trace(go.Bar(
        y=df_plot['DriverCode'],
        x=df_plot['Deviation'],
        orientation='h',
        marker_color=colors,
        name='Prediction Deviation',
        text=[f"{dev:+.1f}" for dev in df_plot['Deviation']],  # Show signed deviation
        textposition='auto',
        textfont=dict(color='white', size=11, family='Arial Black'),
        customdata=df_plot[['Position', 'Ensemble_Pred', 'TeamName', 'AbsError']].values,
        hovertemplate='<b>%{y}</b> (%{customdata[2]})<br>' +
                      'Actual: P%{customdata[0]}<br>' +
                      'Predicted: P%{customdata[1]:.1f}<br>' +
                      'Error: %{customdata[3]:.1f} positions<br>' +
                      'Deviation: %{x:+.1f} positions<extra></extra>'
    ))

    # Add zero reference line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
        line_width=2
    )

    fig.update_layout(
        title={
            'text': "Prediction Accuracy: How Far Off Were Each Prediction?<br>" +
                     "<sub>Green=Accurate, Red=Inaccurate | Positive=Overestimated Finish (predicted worse), Negative=Underestimated (predicted better)</sub>",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 16}
        },
        xaxis_title="Deviation (Predicted Position - Actual Position)",
        yaxis_title="Driver",
        height=max(500, len(df_plot) * 30),  # Taller for better readability
        yaxis=dict(
            autorange='reversed',  # Show biggest errors at top
            tickfont={'size': 12},
            gridcolor='lightgray',
            showgrid=False
        ),
        xaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        ),
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=120, r=20, t=100, b=60),  # Wider left margin for longer title
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.02, y=-0.10,
                text="<b>Color Guide:</b> <span style='color:#00B894'>●</span> Excellent (0-1) | " +
                     "<span style='color:#00CEC9'>●</span> Good (1-2) | " +
                     "<span style='color:#FDCB6E'>●</span> Fair (2-4) | " +
                     "<span style='color:#E17055'>●</span> Poor (4+)",
                showarrow=False,
                font=dict(size=10),
                align='left',
                bgcolor='rgba(248, 249, 250, 0.9)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )

    return fig


def plot_comprehensive_performance_dashboard(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive performance dashboard showing multiple metrics at once.
    
    Args:
        results_df: DataFrame with columns ['DriverCode', 'Position', 'Ensemble_Pred', 'AbsError', 'TeamName', 'GridPosition']

    Returns:
        Plotly figure with comprehensive performance dashboard
    """
    from plotly.subplots import make_subplots
    
    # Calculate key metrics
    mae = results_df['AbsError'].mean()
    median_error = results_df['AbsError'].median()
    perfect_preds = (results_df['AbsError'] < 0.5).sum()
    within_1_pos = (results_df['AbsError'] < 1).sum()
    within_2_pos = (results_df['AbsError'] < 2).sum()
    within_3_pos = (results_df['AbsError'] < 3).sum()
    
    # Calculate baseline comparison (grid position = finish position)
    grid_abs_errors = np.abs(results_df['GridPosition'] - results_df['Position'])
    grid_mae = grid_abs_errors.mean()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Model MAE vs Baseline', 
            'Accuracy Distribution', 
            'Perfect Predictions',
            'Accuracy Within ±2 Pos', 
            'Top 3 Accuracy', 
            'Grid vs Actual Correlation'
        ],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "scatter"}]]
    )
    
    # Model MAE indicator
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=mae,
        domain={'x': [0, 1], 'y': [0.6, 1]},
        title={'text': "Model MAE"},
        delta={'reference': grid_mae, 'relative': False},
        gauge={
            'axis': {'range': [None, 10]},
            'bar': {'color': F1ColorTheme.PREDICTED},
            'steps': [
                {'range': [0, 1], 'color': F1ColorTheme.EXCELLENT},
                {'range': [1, 2], 'color': F1ColorTheme.GOOD},
                {'range': [2, 4], 'color': F1ColorTheme.FAIR},
                {'range': [4, 10], 'color': F1ColorTheme.TERRIBLE}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': mae
            }
        }
    ), row=1, col=1)
    
    # Baseline MAE indicator
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=grid_mae,
        domain={'x': [0, 1], 'y': [0, 0.4]},
        title={'text': "Grid Baseline MAE"},
        gauge={
            'axis': {'range': [None, 10]},
            'bar': {'color': F1ColorTheme.ACTUAL},
            'steps': [
                {'range': [0, 1], 'color': F1ColorTheme.EXCELLENT},
                {'range': [1, 2], 'color': F1ColorTheme.GOOD},
                {'range': [2, 4], 'color': F1ColorTheme.FAIR},
                {'range': [4, 10], 'color': F1ColorTheme.TERRIBLE}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': grid_mae
            }
        }
    ), row=1, col=1)
    
    # Accuracy within thresholds
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=within_2_pos / len(results_df) * 100,
        domain={'x': [0, 1], 'y': [0.6, 1]},
        title={'text': "Within ±2 Pos (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': F1ColorTheme.PREDICTED},
            'steps': [
                {'range': [0, 20], 'color': F1ColorTheme.TERRIBLE},
                {'range': [20, 40], 'color': F1ColorTheme.POOR},
                {'range': [40, 70], 'color': F1ColorTheme.FAIR},
                {'range': [70, 100], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': within_2_pos / len(results_df) * 100
            }
        }
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=within_1_pos / len(results_df) * 100,
        domain={'x': [0, 1], 'y': [0, 0.4]},
        title={'text': "Within ±1 Pos (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': F1ColorTheme.PREDICTED},
            'steps': [
                {'range': [0, 10], 'color': F1ColorTheme.TERRIBLE},
                {'range': [10, 25], 'color': F1ColorTheme.POOR},
                {'range': [25, 50], 'color': F1ColorTheme.FAIR},
                {'range': [50, 100], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': within_1_pos / len(results_df) * 100
            }
        }
    ), row=1, col=2)
    
    # Perfect predictions
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=perfect_preds / len(results_df) * 100,
        domain={'x': [0, 1], 'y': [0.6, 1]},
        title={'text': "Perfect Predictions (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': F1ColorTheme.EXCELLENT},
            'steps': [
                {'range': [0, 5], 'color': F1ColorTheme.TERRIBLE},
                {'range': [5, 15], 'color': F1ColorTheme.POOR},
                {'range': [15, 30], 'color': F1ColorTheme.FAIR},
                {'range': [30, 100], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': perfect_preds / len(results_df) * 100
            }
        }
    ), row=1, col=3)
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=perfect_preds,
        domain={'x': [0, 1], 'y': [0, 0.4]},
        title={'text': "Perfect Predictions (Count)"},
        number={'font': {'color': F1ColorTheme.EXCELLENT, 'size': 24}}
    ), row=1, col=3)
    
    # Within ±2 accuracy
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=within_2_pos,
        domain={'x': [0, 1], 'y': [0.6, 1]},
        title={'text': "Within ±2: Count"},
        gauge={
            'axis': {'range': [None, len(results_df)]},
            'bar': {'color': F1ColorTheme.GOOD},
            'steps': [
                {'range': [0, len(results_df)*0.2], 'color': F1ColorTheme.TERRIBLE},
                {'range': [len(results_df)*0.2, len(results_df)*0.5], 'color': F1ColorTheme.POOR},
                {'range': [len(results_df)*0.5, len(results_df)*0.7], 'color': F1ColorTheme.FAIR},
                {'range': [len(results_df)*0.7, len(results_df)], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': within_2_pos
            }
        }
    ), row=2, col=1)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=within_2_pos / len(results_df) * 100,
        domain={'x': [0, 1], 'y': [0, 0.4]},
        title={'text': "Within ±2: %"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': F1ColorTheme.GOOD},
            'steps': [
                {'range': [0, 20], 'color': F1ColorTheme.TERRIBLE},
                {'range': [20, 50], 'color': F1ColorTheme.POOR},
                {'range': [50, 70], 'color': F1ColorTheme.FAIR},
                {'range': [70, 100], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': within_2_pos / len(results_df) * 100
            }
        }
    ), row=2, col=1)
    
    # Top-3 prediction accuracy
    actual_top_3 = set(results_df.nsmallest(3, 'Position')['DriverCode'])
    predicted_top_3 = set(results_df.nsmallest(3, 'Ensemble_Pred')['DriverCode'])
    top_3_overlap = len(actual_top_3 & predicted_top_3)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=top_3_overlap,
        domain={'x': [0, 1], 'y': [0.6, 1]},
        title={'text': "Top-3 Overlap"},
        gauge={
            'axis': {'range': [None, 3]},
            'bar': {'color': F1ColorTheme.PREDICTED},
            'steps': [
                {'range': [0, 1], 'color': F1ColorTheme.TERRIBLE},
                {'range': [1, 2], 'color': F1ColorTheme.FAIR},
                {'range': [2, 3], 'color': F1ColorTheme.EXCELLENT}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': top_3_overlap
            }
        }
    ), row=2, col=2)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=top_3_overlap / 3 * 100,
        domain={'x': [0, 1], 'y': [0, 0.4]},
        title={'text': "Top-3 Accuracy %"},
        number={'suffix': "%"},
        delta={'reference': 33.33, 'relative': False}  # Random baseline for comparison
    ), row=2, col=2)
    
    # Grid vs Actual scatter for baseline
    fig.add_trace(go.Scatter(
        x=results_df['GridPosition'],
        y=results_df['Position'],
        mode='markers',
        marker=dict(
            size=8,
            color=[F1ColorTheme.get_error_color(err) for err in grid_abs_errors],
            opacity=0.7
        ),
        text=results_df['DriverCode'],
        hovertemplate='<b>%{text}</b><br>Grid: %{x}<br>Actual: %{y}<br>Error: %{marker.color}<extra></extra>',
        name='Grid vs Actual'
    ), row=2, col=3)
    
    # Add y=x line (perfect baseline)
    max_pos = max(results_df['GridPosition'].max(), results_df['Position'].max())
    fig.add_trace(go.Scatter(
        x=[1, max_pos],
        y=[1, max_pos],
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ), row=2, col=3)
    
    fig.update_xaxes(title_text="Grid Position", row=2, col=3)
    fig.update_yaxes(title_text="Actual Position", row=2, col=3)
    
    fig.update_layout(
        title={
            'text': "Comprehensive Performance Dashboard: Model vs Baseline Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=10)
    )
    
    return fig


def plot_position_accuracy_heatmap(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing accuracy by actual vs predicted position ranges.
    
    Args:
        results_df: DataFrame with prediction results

    Returns:
        Plotly figure showing accuracy heatmap
    """
    # Create position bins
    actual_bins = pd.cut(results_df['Position'], bins=range(0, 22, 4), include_lowest=True, labels=['P1-4', 'P5-8', 'P9-12', 'P13-16', 'P17-20'])
    pred_bins = pd.cut(results_df['Ensemble_Pred'], bins=range(0, 22, 4), include_lowest=True, labels=['P1-4', 'P5-8', 'P9-12', 'P13-16', 'P17-20'])
    
    # Calculate accuracy for each bin combination
    accuracy_df = pd.DataFrame(index=['P1-4', 'P5-8', 'P9-12', 'P13-16', 'P17-20'], 
                              columns=['P1-4', 'P5-8', 'P9-12', 'P13-16', 'P17-20'])
    
    for i, actual_bin in enumerate(accuracy_df.index):
        for j, pred_bin in enumerate(accuracy_df.columns):
            mask = (actual_bins == actual_bin) & (pred_bins == pred_bin)
            if mask.any():
                accuracy_df.iloc[i, j] = results_df[mask]['AbsError'].mean()
            else:
                accuracy_df.iloc[i, j] = 0  # or np.nan, depending on preference
    
    fig = go.Figure(data=go.Heatmap(
        z=accuracy_df.values,
        x=accuracy_df.columns,
        y=accuracy_df.index,
        colorscale=[
            [0, F1ColorTheme.EXCELLENT],      # Low error
            [0.5, F1ColorTheme.FAIR],         # Medium error
            [1, F1ColorTheme.TERRIBLE]        # High error
        ],
        hoverongaps=False,
        text=accuracy_df.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate='<b>Actual: %{y}</b><br><b>Predicted: %{x}</b><br>MAE: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Position Accuracy Heatmap: Error by Actual vs Predicted Position Range",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Predicted Position Range",
        yaxis_title="Actual Position Range",
        height=500,
        plot_bgcolor='white',
        font=dict(size=11)
    )
    
    return fig


def add_prediction_methodology_explanation():
    """
    Return markdown content explaining how predictions were made.
    """
    explanation = """
    ### 📊 How These Predictions Were Made
    
    **Timeline**: These predictions were generated **before** the race started using only information available up to that point.
    
    **Methodology**:
    1. **Historical Data**: The model uses patterns from 2022-2024 seasons (excluding this specific race)
    2. **Pre-Race Inputs**: Grid positions, driver performance history, team performance history
    3. **Circuit Context**: Historical performance patterns specific to this circuit type
    4. **ML Processing**: Ensemble model (Random Forest, XGBoost, Linear Regression) weighted combination
    
    **Key Point**: This is **NOT** using the race results to make predictions - these predictions were made in real-time before the race, just like we would for a future race.
    
    **Validation**: The "Past Race Analysis" compares these pre-race predictions to actual results to validate the model's predictive capability for future races.
    """
    return explanation

def plot_comprehensive_performance_dashboard(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive performance dashboard showing multiple metrics at once.
    
    Args:
        results_df: DataFrame with columns ["DriverCode", "Position", "Ensemble_Pred", "AbsError", "TeamName", "GridPosition"]

    Returns:
        Plotly figure with comprehensive performance dashboard
    """
    from plotly.subplots import make_subplots

    # Calculate key metrics
    mae = results_df["AbsError"].mean()
    perfect_preds = (results_df["AbsError"] < 0.5).sum()
    within_1_pos = (results_df["AbsError"] < 1).sum()
    within_2_pos = (results_df["AbsError"] < 2).sum()

    # Calculate baseline comparison (grid position = finish position)
    grid_abs_errors = np.abs(results_df["GridPosition"] - results_df["Position"])
    grid_mae = grid_abs_errors.mean()

    # Top-3 prediction accuracy
    actual_top_3 = set(results_df.nsmallest(3, "Position")["DriverCode"])
    predicted_top_3 = set(results_df.nsmallest(3, "Ensemble_Pred")["DriverCode"])
    top_3_overlap = len(actual_top_3 & predicted_top_3)

    # Create subplots with proper spacing to avoid overlap
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Model MAE", 
            "Grid Baseline MAE", 
            "Within ±2 Pos (%)",
            "Within ±1 Pos (%)", 
            "Perfect Predictions (%)", 
            "Grid vs Actual Correlation"
        ],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "scatter"}]],
        vertical_spacing=0.15,  # Proper spacing to prevent overlap
        horizontal_spacing=0.1
    )

    # Model MAE indicator - single indicator per subplot (no overlap)
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=mae,
        title={"text": "Model MAE"},
        delta={"reference": grid_mae, "relative": False},
        gauge={
            "axis": {"range": [None, 10]},
            "bar": {"color": F1ColorTheme.PREDICTED},
            "steps": [
                {"range": [0, 1], "color": F1ColorTheme.EXCELLENT},
                {"range": [1, 2], "color": F1ColorTheme.GOOD},
                {"range": [2, 4], "color": F1ColorTheme.FAIR},
                {"range": [4, 10], "color": F1ColorTheme.TERRIBLE}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": mae
            }
        }
    ), row=1, col=1)

    # Grid Baseline MAE indicator - single indicator per subplot (no overlap)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=grid_mae,
        title={"text": "Grid Baseline MAE"},
        gauge={
            "axis": {"range": [None, 10]},
            "bar": {"color": F1ColorTheme.ACTUAL},
            "steps": [
                {"range": [0, 1], "color": F1ColorTheme.EXCELLENT},
                {"range": [1, 2], "color": F1ColorTheme.GOOD},
                {"range": [2, 4], "color": F1ColorTheme.FAIR},
                {"range": [4, 10], "color": F1ColorTheme.TERRIBLE}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": grid_mae
            }
        }
    ), row=1, col=2)

    # Within ±2 positions indicator - single indicator per subplot (no overlap)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=within_2_pos / len(results_df) * 100,
        title={"text": "Within ±2 Pos (%)"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": F1ColorTheme.PREDICTED},
            "steps": [
                {"range": [0, 20], "color": F1ColorTheme.TERRIBLE},
                {"range": [20, 40], "color": F1ColorTheme.POOR},
                {"range": [40, 70], "color": F1ColorTheme.FAIR},
                {"range": [70, 100], "color": F1ColorTheme.EXCELLENT}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": within_2_pos / len(results_df) * 100
            }
        }
    ), row=1, col=3)

    # Within ±1 positions indicator - single indicator per subplot (no overlap)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=within_1_pos / len(results_df) * 100,
        title={"text": "Within ±1 Pos (%)"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": F1ColorTheme.PREDICTED},
            "steps": [
                {"range": [0, 10], "color": F1ColorTheme.TERRIBLE},
                {"range": [10, 25], "color": F1ColorTheme.POOR},
                {"range": [25, 50], "color": F1ColorTheme.FAIR},
                {"range": [50, 100], "color": F1ColorTheme.EXCELLENT}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": within_1_pos / len(results_df) * 100
            }
        }
    ), row=2, col=1)

    # Perfect predictions % indicator - single indicator per subplot (no overlap)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=perfect_preds / len(results_df) * 100,
        title={"text": "Perfect Predictions (%)"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": F1ColorTheme.EXCELLENT},
            "steps": [
                {"range": [0, 5], "color": F1ColorTheme.TERRIBLE},
                {"range": [5, 15], "color": F1ColorTheme.POOR},
                {"range": [15, 30], "color": F1ColorTheme.FAIR},
                {"range": [30, 100], "color": F1ColorTheme.EXCELLENT}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": perfect_preds / len(results_df) * 100
            }
        }
    ), row=2, col=2)

    # Grid vs Actual scatter for baseline (in its own subplot since it's different type)
    fig.add_trace(go.Scatter(
        x=results_df["GridPosition"],
        y=results_df["Position"],
        mode="markers",
        marker=dict(
            size=8,
            color=[F1ColorTheme.get_error_color(err) for err in grid_abs_errors],
            opacity=0.7
        ),
        text=results_df["DriverCode"],
        hovertemplate="<b>%{text}</b><br>Grid: %{x}<br>Actual: %{y}<br>Error: %{marker.color}<extra></extra>",
        name="Grid vs Actual"
    ), row=2, col=3)

    # Add y=x line (perfect baseline)
    max_pos = max(results_df["GridPosition"].max(), results_df["Position"].max())
    fig.add_trace(go.Scatter(
        x=[1, max_pos],
        y=[1, max_pos],
        mode="lines",
        line=dict(color="red", dash="dash"),
        showlegend=False
    ), row=2, col=3)

    fig.update_xaxes(title_text="Grid Position", row=2, col=3)
    fig.update_yaxes(title_text="Actual Position", row=2, col=3)

    fig.update_layout(
        title={
            "text": "Comprehensive Performance Dashboard: Model vs Baseline Comparison",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18}
        },
        height=800,
        showlegend=False,
        plot_bgcolor="white",
        font=dict(size=12)
    )

    return fig
