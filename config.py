"""
F1 Race Predictor - Configuration & Constants
Handles team canonical mappings, circuit types, and system constants
"""

# ==========================
# TRAINING CONFIGURATION
# ==========================
TRAINING_SEASONS = [2022, 2023, 2024]  # Ground-effect era only
MIN_DRIVER_RACES = 5  # Threshold for "rookie" treatment
MIN_CIRCUIT_RACES = 2  # Threshold for circuit fallback
CACHE_DIR = './f1_cache'  # FastF1 cache location

# ==========================
# TEAM CANONICAL MAPPING
# ==========================
# Maps all team name variations to canonical team identifiers
# This handles team renames, ownership changes, and data consistency
TEAM_CANONICAL_MAP = {
    # Aston Martin lineage
    'Aston Martin': 'Aston Martin',
    'Aston Martin Aramco Cognizant F1 Team': 'Aston Martin',
    'Aston Martin Aramco Mercedes': 'Aston Martin',
    'Racing Point': 'Aston Martin',  # 2020 predecessor
    'Force India': 'Aston Martin',  # Historical predecessor
    
    # Alpine/Renault lineage
    'Alpine F1 Team': 'Alpine',
    'Alpine': 'Alpine',
    'Renault': 'Alpine',  # Pre-2021 name
    'BWT Alpine F1 Team': 'Alpine',
    
    # Sauber lineage (multiple identity changes)
    'Alfa Romeo': 'Sauber Group',
    'Alfa Romeo Racing': 'Sauber Group',
    'Alfa Romeo Racing ORLEN': 'Sauber Group',
    'Sauber': 'Sauber Group',
    'Stake F1 Team': 'Sauber Group',
    'Kick Sauber F1 Team': 'Sauber Group',
    'Stake F1 Team Kick Sauber': 'Sauber Group',
    
    # AlphaTauri/RB lineage
    'AlphaTauri': 'RB',
    'Scuderia AlphaTauri': 'RB',
    'RB F1 Team': 'RB',
    'RB': 'RB',
    'Racing Bulls': 'RB',
    'Visa Cash App RB F1 Team': 'RB',
    
    # Stable teams (no major identity changes in this era)
    'Red Bull Racing': 'Red Bull Racing',
    'Red Bull Racing Honda RBPT': 'Red Bull Racing',
    'Oracle Red Bull Racing': 'Red Bull Racing',
    
    'Ferrari': 'Ferrari',
    'Scuderia Ferrari': 'Ferrari',
    'Scuderia Ferrari HP': 'Ferrari',
    
    'Mercedes': 'Mercedes',
    'Mercedes-AMG Petronas F1 Team': 'Mercedes',
    'Mercedes-AMG PETRONAS': 'Mercedes',
    
    'McLaren': 'McLaren',
    'McLaren Racing': 'McLaren',
    'McLaren F1 Team': 'McLaren',
    'McLaren Mercedes': 'McLaren',
    
    'Williams': 'Williams',
    'Williams Racing': 'Williams',
    'Williams Mercedes': 'Williams',
    
    'Haas F1 Team': 'Haas',
    'Haas': 'Haas',
    'MoneyGram Haas F1 Team': 'Haas',
}

# ==========================
# CIRCUIT TYPE MAPPING
# ==========================
# Used for fallback when circuit has insufficient historical data
CIRCUIT_TYPES = {
    # Street circuits (high variance, more unpredictable)
    'Monaco': 'street',
    'Marina Bay': 'street',  # Singapore
    'Baku': 'street',
    'Jeddah': 'street',
    'Las Vegas': 'street',
    'Miami': 'street',
    
    # Permanent circuits (lower variance, more predictable)
    'Albert Park': 'permanent',  # Australia
    'Bahrain International Circuit': 'permanent',
    'Shanghai International Circuit': 'permanent',
    'Suzuka': 'permanent',
    'Circuit de Barcelona-Catalunya': 'permanent',
    'Circuit de Monaco': 'street',  # Duplicate, kept for safety
    'Circuit Gilles Villeneuve': 'hybrid',  # Montreal (semi-street)
    'Red Bull Ring': 'permanent',  # Austria
    'Silverstone': 'permanent',
    'Hungaroring': 'permanent',
    'Spa-Francorchamps': 'permanent',
    'Zandvoort': 'permanent',
    'Monza': 'permanent',
    'Circuit of the Americas': 'permanent',  # Austin
    'Autódromo Hermanos Rodríguez': 'permanent',  # Mexico
    'Interlagos': 'permanent',  # Brazil
    'Yas Marina Circuit': 'permanent',  # Abu Dhabi
    'Losail': 'permanent',  # Qatar
    'Imola': 'permanent',
}

# ==========================
# FEATURE COLUMN DEFINITION
# ==========================
# All features used in model training (order matters for consistency)
FEATURE_COLUMNS = [
    # Race identifiers
    'Year',
    'CircuitEnc',
    'RoundNumber',
    
    # Grid & context
    'GridPosition',
    'IsWetRace',
    'IsSprintWeekend',
    
    # Driver features
    'DriverEnc',
    'DriverTotalRaces',
    'DriverAvgPos',
    'DriverBestPos',
    'DriverStd',
    'DriverDNFRate',
    'DriverAvgGrid',
    
    # Team features
    'TeamEnc',
    'TeamTotalRaces',
    'TeamAvgPos',
    'TeamStd',
    'TeamDNFRate',
    
    # Circuit features
    'CircuitPositionVariance',
    'CircuitDNFRate',
    'CircuitRaceCount',
    
    # Engineered features
    'GridAdvantage',
    'TeamReliability',
    'DriverForm',
    'IsStreetCircuit',
    'SeasonProgressFraction',
]

# ==========================
# POINTS SYSTEM
# ==========================
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

# ==========================
# MODEL HYPERPARAMETERS
# ==========================
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'linear': {
        'fit_intercept': True
    }
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'rf': 0.4,
    'xgb': 0.4,
    'lr': 0.2
}

# ==========================
# SPRINT WEEKEND MAPPING
# ==========================
# Manual mapping of sprint weekends (FastF1 detection can be unreliable)
# Format: {year: [event_names]}
SPRINT_WEEKENDS = {
    2022: ['Emilia Romagna', 'Austria', 'São Paulo'],
    2023: ['Azerbaijan', 'Austria', 'Belgium', 'Qatar', 'United States', 'São Paulo'],
    2024: ['China', 'Miami', 'Austria', 'United States', 'São Paulo', 'Qatar'],
    2025: ['China', 'Miami', 'Belgium', 'United States', 'São Paulo', 'Qatar'],
}

# ==========================
# UI CONFIGURATION
# ==========================
STREAMLIT_CONFIG = {
    'page_title': 'F1 Race Outcome Predictor',
    'page_icon': '🏎️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color scheme for visualizations
POSITION_COLORS = {
    'excellent': '#00D856',  # Positions 1-3
    'good': '#FFB84D',       # Positions 4-7
    'average': '#FF6B6B',    # Positions 8-10
    'poor': '#A0A0A0'        # Positions 11+
}