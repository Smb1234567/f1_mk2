"""
DEPRECATED: This file exists for backward compatibility.
All visualization functionality has been moved to the `visuals` package.
Please update your imports to use `from visuals import ...` instead.
"""

# Import everything from the new package for backward compatibility
from visuals import (
    F1ColorTheme,
    calculate_summary_metrics,
    plot_summary_metrics,
    plot_error_distribution,
    plot_dumbbell_positions,
    plot_error_by_driver,
    plot_actual_vs_predicted_scatter,
    plot_predicted_order,
    plot_gain_loss_vs_grid,
    plot_team_performance_comparison
)

# For backward compatibility - map old function names to new ones
create_summary_metrics = calculate_summary_metrics
plot_hero_metrics = plot_summary_metrics

__all__ = [
    'F1ColorTheme',
    'create_summary_metrics',  # Backward compatibility
    'calculate_summary_metrics',
    'plot_hero_metrics',       # Backward compatibility
    'plot_summary_metrics',
    'plot_error_distribution',
    'plot_dumbbell_positions',
    'plot_error_by_driver',
    'plot_actual_vs_predicted_scatter',
    'plot_predicted_order',
    'plot_gain_loss_vs_grid',
    'plot_team_performance_comparison'
]


# ============================================================================
# ERROR BY DRIVER - WHO'S HARD TO PREDICT? (FIXED)
# ============================================================================

def plot_error_by_driver(results_df: pd.DataFrame) -> go.Figure:
    """
    IMPROVED: Clear horizontal bar chart showing prediction reliability
    
    FIXES:
    - Consistent color theming
    - Better sorting (worst predictions at top for visibility)
    - Racing-context annotations
    
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
            'text': "Prediction Accuracy by Driver: Who's Easy vs Hard to Predict?",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Prediction Error (Positions)",
        yaxis_title="Driver",
        height=max(420, len(df_sorted) * 26),
        yaxis=dict(
            # Show worst predictions at top for visibility
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
        margin=dict(l=80, r=20, t=70, b=60),
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.02, y=-0.10,
                text="<b>Color Guide:</b> <span style='color:#00B894'>●</span> Excellent (0-1) | " +
                     "<span style='color:#00CEC9'>●</span> Good (1-2) | " +
                     "<span style='color:#FDCB6E'>●</span> Fair (2-4) | " +
                     "<span style='color:#E17055'>●</span> Poor (4-6) | " +
                     "<span style='color:#D63031'>●</span> Very Poor (6+)",
                showarrow=False,
                font=dict(size=11),
                align='left',
                bgcolor='rgba(248, 249, 250, 0.9)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )
    
    return fig


# ============================================================================
# SCATTER PLOT - MODEL PERFORMANCE ASSESSMENT (FIXED)
# ============================================================================

def plot_actual_vs_predicted_scatter(results_df: pd.DataFrame) -> go.Figure:
    """
    IMPROVED scatter plot with FIXED hover template and clear interpretation
    
    FIXES:
    - Corrected hover template calculation (was showing same value twice!)
    - Standard axis orientation (no reversal - confusing for beginners)
    - Clear performance zones
    
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
        # FIXED: Corrected hover template calculation
        hovertemplate='<b>%{text}</b> (%{customdata[2]})<br>' +
                      'Actual: P%{customdata[0]}<br>' +
                      'Predicted: P%{customdata[1]:.1f}<br>' +
                      'Error: %{customdata[3]:.1f} positions<extra></extra>',
        name='Drivers'
    ))
    
    fig.update_layout(
        title={
            'text': "Model Performance: Predictions vs Reality",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Actual Position",
        yaxis_title="Predicted Position",
        xaxis=dict(
            # FIXED: Standard orientation (no reversal for scatter plots)
            dtick=2,
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5
        ),
        yaxis=dict(
            # FIXED: Standard orientation
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
        margin=dict(l=60, r=20, t=80, b=60),
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text="💡 <b>Points on red line = perfect predictions</b><br>" +
                     "Points above line = predicted worse than actual<br>" +
                     "Points below line = predicted better than actual",
                showarrow=False,
                font=dict(size=11, color='#2D3436'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                align='left'
            )
        ]
    )
    
    return fig


# ============================================================================
# PREDICTED ORDER - FUTURE RACE VISUALIZATION (IMPROVED)
# ============================================================================

def plot_predicted_order(result: pd.DataFrame) -> go.Figure:
    """
    IMPROVED predicted finishing order with better team differentiation
    
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
            'text': "Predicted Race Finish: The Expected Order",
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 18}
        },
        xaxis_title="Predicted Position (Lower = Better)",
        yaxis_title="Driver",
        height=max(450, len(df_sorted) * 28),
        yaxis=dict(
            # Show best predictions at top
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


# ============================================================================
# GAIN/LOSS FROM GRID - EXPECTED MOVERS (IMPROVED)
# ============================================================================

def plot_gain_loss_vs_grid(result: pd.DataFrame) -> go.Figure:
    """
    IMPROVED diverging bar chart showing expected position changes
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
            'text': "Expected Position Changes: Who Will Move Up or Down?",
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
        margin=dict(l=60, r=20, t=70, b=100),
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.02, y=-0.15,
                text="<b>Color Guide:</b> <span style='color:#00B894'>●</span> Big Gain (3+) | " +
                     "<span style='color:#55EFC4'>●</span> Small Gain | " +
                     "<span style='color:#A29BFE'>●</span> No Change | " +
                     "<span style='color:#FDCB6E'>●</span> Small Loss | " +
                     "<span style='color:#E17055'>●</span> Big Loss (3+)",
                showarrow=False,
                font=dict(size=11),
                align='left',
                bgcolor='rgba(248, 249, 250, 0.9)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )
    
    return fig


# ============================================================================
# TEAM PERFORMANCE COMPARISON (IMPROVED)
# ============================================================================

def plot_team_performance_comparison(results_df: pd.DataFrame) -> go.Figure:
    """
    IMPROVED team-level analysis showing actual vs predicted performance
    
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