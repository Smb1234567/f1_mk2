"""
F1 Visualization Package
"""

from .color_theme import F1ColorTheme
from .metrics_calculator import calculate_summary_metrics
from .chart_generators import (
    plot_summary_metrics,
    plot_error_distribution,
    plot_dumbbell_positions,
    plot_error_by_driver,
    plot_actual_vs_predicted_scatter,
    plot_predicted_order,
    plot_gain_loss_vs_grid,
    plot_team_performance_comparison,
    plot_prediction_deviation_stories,
    plot_comprehensive_performance_dashboard,
    plot_position_accuracy_heatmap,
    add_prediction_methodology_explanation
)
from .enhanced_future_visuals import (
    plot_model_confidence_metrics,
    plot_position_changes_story,
    plot_prediction_certainty_bands,
    plot_team_performance_story,
    highlight_key_predictions
)

__all__ = [
    'F1ColorTheme',
    'calculate_summary_metrics',
    'plot_summary_metrics',
    'plot_error_distribution',
    'plot_dumbbell_positions',
    'plot_error_by_driver',
    'plot_actual_vs_predicted_scatter',
    'plot_predicted_order',
    'plot_gain_loss_vs_grid',
    'plot_team_performance_comparison',
    'plot_prediction_deviation_stories',
    'plot_comprehensive_performance_dashboard',
    'plot_position_accuracy_heatmap',
    'add_prediction_methodology_explanation',
    'plot_model_confidence_metrics',
    'plot_position_changes_story',
    'plot_prediction_certainty_bands',
    'plot_team_performance_story',
    'highlight_key_predictions'
]