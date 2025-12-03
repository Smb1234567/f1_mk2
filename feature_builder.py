"""
F1 Race Predictor - Feature Engineering Module
Handles all feature computation with comprehensive edge case handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from config import (
    TEAM_CANONICAL_MAP, CIRCUIT_TYPES, SPRINT_WEEKENDS,
    MIN_DRIVER_RACES, MIN_CIRCUIT_RACES, FEATURE_COLUMNS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Transforms raw race data into ML-ready features.
    
    Handles critical edge cases:
    - Rookie drivers (low race count)
    - Team changes/renames
    - New circuits
    - Missing data
    """
    
    def __init__(self, raw_data: pd.DataFrame):
        """
        Initialize with raw race results.

        Args:
            raw_data: DataFrame from data_loader with race results
        """
        self.raw_data = raw_data.copy()

        # Only apply canonical team mapping if the DataFrame has data
        if not raw_data.empty and 'TeamName' in raw_data.columns:
            # Apply canonical team mapping immediately
            self.raw_data['TeamCanonical'] = self.raw_data['TeamName'].map(
                lambda x: TEAM_CANONICAL_MAP.get(x, x)
            )

        # Computed statistics (filled by compute_stats methods)
        self.driver_stats = None
        self.team_stats = None
        self.circuit_stats = None

        # Encoding maps (for consistent string->int conversion)
        self.driver_map = {}
        self.team_map = {}
        self.circuit_map = {}

        logger.info(f"FeatureBuilder initialized with {len(raw_data)} rows")
    
    def compute_driver_stats(self) -> pd.DataFrame:
        """
        Compute per-driver statistics across all their races.
        
        Returns DataFrame with:
        - DriverCode: Driver identifier
        - DriverTotalRaces: Count of races completed
        - DriverAvgPos: Average finishing position
        - DriverBestPos: Best finishing position
        - DriverStd: Std dev of positions
        - DriverDNFRate: DNF rate (proportion)
        - DriverAvgGrid: Average starting position
        """
        # Filter out DNFs for position stats (keep them for DNF rate)
        # DNF = Position is NaN or Status contains 'Retired', 'Accident', etc.
        df = self.raw_data.copy()
        df['IsDNF'] = df['Position'].isna() | df['Status'].str.contains(
            'Retired|Accident|Collision|Damage|Mechanical', 
            case=False, 
            na=False
        )
        
        stats = []
        for driver in df['DriverCode'].unique():
            driver_df = df[df['DriverCode'] == driver]
            finished_df = driver_df[~driver_df['IsDNF']]
            
            total_races = len(driver_df)
            dnf_count = driver_df['IsDNF'].sum()
            
            # Position stats (only from finished races)
            if len(finished_df) > 0:
                avg_pos = finished_df['Position'].mean()
                best_pos = finished_df['Position'].min()
                std_pos = finished_df['Position'].std() if len(finished_df) > 1 else 0
            else:
                # Driver has DNF'd in all races - use defaults
                avg_pos = 15.0  # Pessimistic default
                best_pos = 20.0
                std_pos = 0.0
            
            # Grid stats (from all races)
            avg_grid = driver_df['GridPosition'].mean()
            if pd.isna(avg_grid):
                avg_grid = 15.0  # Default midfield
            
            stats.append({
                'DriverCode': driver,
                'DriverTotalRaces': total_races,
                'DriverAvgPos': avg_pos,
                'DriverBestPos': best_pos,
                'DriverStd': std_pos,
                'DriverDNFRate': dnf_count / max(total_races, 1),
                'DriverAvgGrid': avg_grid,
            })
        
        self.driver_stats = pd.DataFrame(stats)
        logger.info(f"Computed stats for {len(self.driver_stats)} drivers")
        return self.driver_stats
    
    def compute_team_stats(self) -> pd.DataFrame:
        """
        Compute per-team statistics (using canonical team names).
        
        IMPORTANT: Uses TeamCanonical to aggregate across team renames.
        """
        df = self.raw_data.copy()
        df['IsDNF'] = df['Position'].isna() | df['Status'].str.contains(
            'Retired|Accident|Collision|Damage|Mechanical', 
            case=False, 
            na=False
        )
        
        stats = []
        for team in df['TeamCanonical'].unique():
            team_df = df[df['TeamCanonical'] == team]
            finished_df = team_df[~team_df['IsDNF']]
            
            total_races = len(team_df)
            dnf_count = team_df['IsDNF'].sum()
            
            if len(finished_df) > 0:
                avg_pos = finished_df['Position'].mean()
                std_pos = finished_df['Position'].std() if len(finished_df) > 1 else 0
            else:
                avg_pos = 12.0  # Default
                std_pos = 0.0
            
            stats.append({
                'TeamCanonical': team,
                'TeamTotalRaces': total_races,
                'TeamAvgPos': avg_pos,
                'TeamStd': std_pos,
                'TeamDNFRate': dnf_count / max(total_races, 1),
            })
        
        self.team_stats = pd.DataFrame(stats)
        logger.info(f"Computed stats for {len(self.team_stats)} teams")
        return self.team_stats
    
    def compute_circuit_stats(self) -> pd.DataFrame:
        """
        Compute per-circuit statistics.
        
        Circuit stats measure track characteristics:
        - Position variance (how much grid position changes during race)
        - DNF rate (reliability/safety challenge)
        - Race count (data availability)
        """
        df = self.raw_data.copy()
        df['IsDNF'] = df['Position'].isna() | df['Status'].str.contains(
            'Retired|Accident|Collision|Damage|Mechanical', 
            case=False, 
            na=False
        )
        
        stats = []
        for circuit in df['CircuitName'].unique():
            circuit_df = df[df['CircuitName'] == circuit]
            finished_df = circuit_df[~circuit_df['IsDNF']]
            
            # Position variance: measure of unpredictability
            # = variance of (Position - GridPosition) for finished races
            if len(finished_df) > 0:
                finished_df_clean = finished_df.dropna(subset=['Position', 'GridPosition'])
                if len(finished_df_clean) > 0:
                    position_changes = finished_df_clean['Position'] - finished_df_clean['GridPosition']
                    pos_variance = position_changes.var()
                else:
                    pos_variance = 5.0  # Default moderate variance
            else:
                pos_variance = 5.0
            
            # DNF rate
            total_starters = len(circuit_df)
            dnf_count = circuit_df['IsDNF'].sum()
            dnf_rate = dnf_count / max(total_starters, 1)
            
            # Count unique race events (not driver entries)
            race_count = circuit_df.groupby(['Year', 'RoundNumber']).ngroups
            
            stats.append({
                'CircuitName': circuit,
                'CircuitPositionVariance': pos_variance,
                'CircuitDNFRate': dnf_rate,
                'CircuitRaceCount': race_count,
            })
        
        self.circuit_stats = pd.DataFrame(stats)
        logger.info(f"Computed stats for {len(self.circuit_stats)} circuits")
        return self.circuit_stats
    
    def create_encoding_maps(self):
        """Create stable string->int encoding for categorical features."""
        # Driver encoding
        drivers = sorted(self.raw_data['DriverCode'].unique())
        self.driver_map = {driver: idx for idx, driver in enumerate(drivers)}
        
        # Team encoding (canonical names)
        teams = sorted(self.raw_data['TeamCanonical'].unique())
        self.team_map = {team: idx for idx, team in enumerate(teams)}
        
        # Circuit encoding
        circuits = sorted(self.raw_data['CircuitName'].unique())
        self.circuit_map = {circuit: idx for idx, circuit in enumerate(circuits)}
        
        logger.info(f"Created encodings: {len(drivers)} drivers, {len(teams)} teams, {len(circuits)} circuits")
    
    def handle_rookie_driver(self, driver_code: str, team_canonical: str) -> Dict:
        """
        Handle low-data drivers (rookies, substitutes).
        
        Strategy: Blend team performance with global driver averages.
        
        Args:
            driver_code: Driver identifier
            team_canonical: Current team (canonical name)
        
        Returns:
            Dict with adjusted driver stats
        """
        # Get global averages
        global_avg_pos = self.driver_stats['DriverAvgPos'].mean()
        global_avg_grid = self.driver_stats['DriverAvgGrid'].mean()
        global_dnf_rate = self.driver_stats['DriverDNFRate'].mean()
        
        # Get team stats for this driver's team
        team_row = self.team_stats[self.team_stats['TeamCanonical'] == team_canonical]
        if len(team_row) > 0:
            team_avg_pos = team_row['TeamAvgPos'].iloc[0]
            team_dnf_rate = team_row['TeamDNFRate'].iloc[0]
        else:
            team_avg_pos = global_avg_pos
            team_dnf_rate = global_dnf_rate
        
        # Blend: 70% team, 30% global (assumption: car matters more than driver skill initially)
        adjusted_avg_pos = 0.7 * team_avg_pos + 0.3 * global_avg_pos
        adjusted_dnf_rate = 0.7 * team_dnf_rate + 0.3 * global_dnf_rate
        adjusted_avg_grid = global_avg_grid  # No team effect on qualifying initially
        
        return {
            'DriverTotalRaces': 0,  # Flag as low-data
            'DriverAvgPos': adjusted_avg_pos,
            'DriverBestPos': adjusted_avg_pos,  # Conservative: no standout performances yet
            'DriverStd': global_avg_pos * 0.2,  # Moderate uncertainty
            'DriverDNFRate': adjusted_dnf_rate,
            'DriverAvgGrid': adjusted_avg_grid,
        }
    
    def handle_new_circuit(self, circuit_name: str) -> Dict:
        """
        Handle circuits with insufficient historical data.
        
        Strategy: Use circuit type (street/permanent) averages as fallback.
        """
        circuit_type = CIRCUIT_TYPES.get(circuit_name, 'permanent')
        
        # Get averages for this circuit type
        circuits_of_type = [
            c for c in self.circuit_stats['CircuitName']
            if CIRCUIT_TYPES.get(c, 'permanent') == circuit_type
        ]
        
        if circuits_of_type:
            type_stats = self.circuit_stats[
                self.circuit_stats['CircuitName'].isin(circuits_of_type)
            ]
            avg_variance = type_stats['CircuitPositionVariance'].mean()
            avg_dnf_rate = type_stats['CircuitDNFRate'].mean()
        else:
            # Global fallback
            avg_variance = self.circuit_stats['CircuitPositionVariance'].mean()
            avg_dnf_rate = self.circuit_stats['CircuitDNFRate'].mean()
        
        return {
            'CircuitPositionVariance': avg_variance,
            'CircuitDNFRate': avg_dnf_rate,
            'CircuitRaceCount': 0,  # Flag as low-data
        }
    
    def build_features(self, target_race_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build feature matrix from race data.
        
        Args:
            target_race_data: Specific race to build features for (e.g., for prediction).
                              If None, builds features for all races in raw_data.
        
        Returns:
            DataFrame with FEATURE_COLUMNS ready for ML models
        """
        # Ensure stats are computed
        if self.driver_stats is None:
            self.compute_driver_stats()
        if self.team_stats is None:
            self.compute_team_stats()
        if self.circuit_stats is None:
            self.compute_circuit_stats()
        if not self.driver_map:
            self.create_encoding_maps()
        
        # Determine which data to build features for
        if target_race_data is not None:
            df = target_race_data.copy()
        else:
            df = self.raw_data.copy()
        
        # Apply canonical team mapping
        if 'TeamCanonical' not in df.columns and 'TeamName' in df.columns:
            df['TeamCanonical'] = df['TeamName'].map(
                lambda x: TEAM_CANONICAL_MAP.get(x, x)
            )
        
        features_list = []
        
        for idx, row in df.iterrows():
            driver = row['DriverCode']
            team = row['TeamCanonical']
            circuit = row['CircuitName']
            year = row['Year']
            round_num = row.get('RoundNumber', 0)
            grid_pos = row['GridPosition']
            
            # Handle missing grid position
            if pd.isna(grid_pos):
                grid_pos = 15.0  # Default midfield
            
            # Get driver stats (with rookie fallback)
            driver_row = self.driver_stats[self.driver_stats['DriverCode'] == driver]
            if len(driver_row) > 0 and driver_row['DriverTotalRaces'].iloc[0] >= MIN_DRIVER_RACES:
                driver_feats = driver_row.iloc[0].to_dict()
            else:
                driver_feats = self.handle_rookie_driver(driver, team)
            
            # Get team stats
            team_row = self.team_stats[self.team_stats['TeamCanonical'] == team]
            if len(team_row) > 0:
                team_feats = team_row.iloc[0].to_dict()
            else:
                # New team fallback (rare)
                team_feats = {
                    'TeamTotalRaces': 0,
                    'TeamAvgPos': 12.0,
                    'TeamStd': 3.0,
                    'TeamDNFRate': 0.15,
                }
            
            # Get circuit stats (with new circuit fallback)
            circuit_row = self.circuit_stats[self.circuit_stats['CircuitName'] == circuit]
            if len(circuit_row) > 0 and circuit_row['CircuitRaceCount'].iloc[0] >= MIN_CIRCUIT_RACES:
                circuit_feats = circuit_row.iloc[0].to_dict()
            else:
                circuit_feats = self.handle_new_circuit(circuit)
            
            # Sprint weekend detection
            event_name = row.get('EventName', '')
            is_sprint = int(any(
                event_name in SPRINT_WEEKENDS.get(year, [])
                for year in SPRINT_WEEKENDS.keys()
            ))
            
            # Build feature dict
            features = {
                'Year': year,
                'CircuitEnc': self.circuit_map.get(circuit, -1),
                'RoundNumber': round_num,
                'GridPosition': grid_pos,
                'IsWetRace': row.get('IsWetRace', 0),
                'IsSprintWeekend': is_sprint,
                
                'DriverEnc': self.driver_map.get(driver, -1),
                'DriverTotalRaces': driver_feats['DriverTotalRaces'],
                'DriverAvgPos': driver_feats['DriverAvgPos'],
                'DriverBestPos': driver_feats['DriverBestPos'],
                'DriverStd': driver_feats['DriverStd'],
                'DriverDNFRate': driver_feats['DriverDNFRate'],
                'DriverAvgGrid': driver_feats['DriverAvgGrid'],
                
                'TeamEnc': self.team_map.get(team, -1),
                'TeamTotalRaces': team_feats['TeamTotalRaces'],
                'TeamAvgPos': team_feats['TeamAvgPos'],
                'TeamStd': team_feats['TeamStd'],
                'TeamDNFRate': team_feats['TeamDNFRate'],
                
                'CircuitPositionVariance': circuit_feats['CircuitPositionVariance'],
                'CircuitDNFRate': circuit_feats['CircuitDNFRate'],
                'CircuitRaceCount': circuit_feats['CircuitRaceCount'],
                
                # Engineered features
                'GridAdvantage': driver_feats['DriverAvgPos'] - grid_pos,
                'TeamReliability': 1 - team_feats['TeamDNFRate'],
                'DriverForm': driver_feats['DriverAvgPos'] / max(driver_feats['DriverBestPos'], 1),
                'IsStreetCircuit': int(CIRCUIT_TYPES.get(circuit, 'permanent') == 'street'),
                'SeasonProgressFraction': round_num / 24.0,  # Assuming 24 races
            }
            
            # Add target if available
            if 'Position' in row and not pd.isna(row['Position']):
                features['ActualPosition'] = row['Position']
            
            # Add identifiers for later reference
            features['DriverCode'] = driver
            features['TeamName'] = row['TeamName']
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Built features for {len(features_df)} driver-race combinations")
        
        return features_df


# ==========================
# UTILITY FUNCTIONS
# ==========================

def identify_edge_cases(features_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify which drivers/circuits used fallback stats.
    
    Returns dict with:
    - rookie_drivers: List of drivers with low race count
    - new_circuits: List of circuits with low race count
    """
    edge_cases = {
        'rookie_drivers': [],
        'new_circuits': [],
    }
    
    # Find rookies
    rookies = features_df[features_df['DriverTotalRaces'] < MIN_DRIVER_RACES]['DriverCode'].unique()
    edge_cases['rookie_drivers'] = list(rookies)
    
    # Find new circuits
    new_circuits = features_df[features_df['CircuitRaceCount'] < MIN_CIRCUIT_RACES]['CircuitName'].unique()
    # Note: CircuitName not in features_df directly, need to track separately
    
    return edge_cases


# ==========================
# EXAMPLE USAGE
# ==========================

if __name__ == '__main__':
    from data_loader import load_raw_data
    
    # Load previously fetched data
    raw_data = load_raw_data('raw_race_data.csv')
    
    # Build features
    builder = FeatureBuilder(raw_data)
    features = builder.build_features()
    
    print("\nFeature matrix shape:", features.shape)
    print("\nSample features:")
    print(features[FEATURE_COLUMNS].head())
    
    print("\nFeature statistics:")
    print(features[FEATURE_COLUMNS].describe())
    
    # Check for edge cases
    print(f"\nDrivers with < {MIN_DRIVER_RACES} races:")
    print(features[features['DriverTotalRaces'] < MIN_DRIVER_RACES]['DriverCode'].unique())
    
    print(f"\nCircuits with < {MIN_CIRCUIT_RACES} races:")
    print(features[features['CircuitRaceCount'] < MIN_CIRCUIT_RACES].groupby('CircuitEnc').size())