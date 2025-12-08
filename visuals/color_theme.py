"""
Consistent color scheme for all F1 visualizations
"""

class F1ColorTheme:
    """Consistent color scheme for all F1 visualizations"""
    
    # Performance accuracy colors (for prediction errors)
    EXCELLENT = '#00B894'    # Green - error < 1 position
    GOOD = '#00CEC9'         # Cyan - error 1-2 positions
    FAIR = '#FDCB6E'         # Yellow - error 2-4 positions
    POOR = '#E17055'         # Orange - error 4-6 positions
    TERRIBLE = '#D63031'     # Red - error > 6 positions
    
    # Position change colors (gains/losses)
    BIG_GAIN = '#00B894'     # Dark green - gained 3+ positions
    SMALL_GAIN = '#55EFC4'   # Light green - gained 0.5-3 positions
    NO_CHANGE = '#A29BFE'    # Purple - minimal change
    SMALL_LOSS = '#FDCB6E'   # Yellow - lost 0.5-3 positions
    BIG_LOSS = '#E17055'     # Orange - lost 3+ positions
    
    # Chart element colors
    ACTUAL = '#0984E3'       # Blue - actual results
    PREDICTED = '#FD79A8'    # Pink - predictions
    REFERENCE = '#2D3436'    # Dark gray - reference lines
    BACKGROUND = '#F8F9FA'   # Light gray background
    
    @staticmethod
    def get_error_color(error: float) -> str:
        """Get color based on prediction error magnitude"""
        if error < 1:
            return F1ColorTheme.EXCELLENT
        elif error < 2:
            return F1ColorTheme.GOOD
        elif error < 4:
            return F1ColorTheme.FAIR
        elif error < 6:
            return F1ColorTheme.POOR
        else:
            return F1ColorTheme.TERRIBLE
    
    @staticmethod
    def get_delta_color(delta: float) -> str:
        """Get color based on position change (positive = gained positions)"""
        if delta > 3:
            return F1ColorTheme.BIG_GAIN
        elif delta > 0.5:
            return F1ColorTheme.SMALL_GAIN
        elif delta < -3:
            return F1ColorTheme.BIG_LOSS
        elif delta < -0.5:
            return F1ColorTheme.SMALL_LOSS
        else:
            return F1ColorTheme.NO_CHANGE