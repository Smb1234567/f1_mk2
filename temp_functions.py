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
        text=np.round(accuracy_df.values, 2),
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