"""F1 Race Predictor - Data Loader Module
Wraps FastF1 API with comprehensive error handling and caching
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging
import warnings

# Suppress FastF1 warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_tracker import RaceLoadingTracker


class F1DataLoader:
    """
    Handles all FastF1 data fetching with robust error handling.

    CRITICAL: FastF1 can fail for various reasons (server issues, missing data,
    recent sessions not yet available). This class handles all failures gracefully.
    """

    def __init__(self, cache_dir: str = './f1_cache'):
        """Initialize data loader with caching enabled."""
        self.cache_dir = cache_dir
        fastf1.Cache.enable_cache(cache_dir)
        logger.info(f"FastF1 cache enabled at: {cache_dir}")
        self.tracker = RaceLoadingTracker()
    
    def get_available_seasons(self, start_year: int = 2022, end_year: int = 2025) -> List[int]:
        """
        Get list of seasons we can attempt to load.
        
        Note: Just because a season is listed doesn't mean all its data will load.
        """
        return list(range(start_year, end_year + 1))
    
    def get_event_schedule(self, year: int) -> Optional[pd.DataFrame]:
        """
        Get the event schedule for a given year.
        
        Returns:
            DataFrame with event information, or None if failed
        """
        try:
            schedule = fastf1.get_event_schedule(year)
            logger.info(f"Loaded schedule for {year}: {len(schedule)} events")
            return schedule
        except Exception as e:
            logger.warning(f"Failed to load schedule for {year}: {e}")
            return None
    
    def get_completed_events(self, year: int) -> List[str]:
        """
        Get list of completed race events for a given year.
        
        Filters out:
        - Testing sessions
        - Future events
        - Events without race sessions
        
        Returns:
            List of event names (e.g., ['Bahrain', 'Saudi Arabia', ...])
        """
        schedule = self.get_event_schedule(year)
        if schedule is None:
            return []
        
        completed = []
        now = datetime.now()
        
        for idx, event in schedule.iterrows():
            # Skip testing
            if 'Testing' in event.get('EventName', ''):
                continue
            
            # Check if event has passed
            event_date = event.get('EventDate')
            if pd.isna(event_date):
                continue
                
            if isinstance(event_date, str):
                event_date = pd.to_datetime(event_date)
            
            if event_date < pd.Timestamp(now):
                completed.append(event['EventName'])
        
        logger.info(f"{year}: Found {len(completed)} completed events")
        return completed
    
    def get_upcoming_events(self, year: int) -> List[str]:
        """Get list of upcoming (future) race events."""
        schedule = self.get_event_schedule(year)
        if schedule is None:
            return []
        
        upcoming = []
        now = datetime.now()
        
        for idx, event in schedule.iterrows():
            if 'Testing' in event.get('EventName', ''):
                continue
            
            event_date = event.get('EventDate')
            if pd.isna(event_date):
                continue
                
            if isinstance(event_date, str):
                event_date = pd.to_datetime(event_date)
            
            if event_date > pd.Timestamp(now):
                upcoming.append(event['EventName'])
        
        logger.info(f"{year}: Found {len(upcoming)} upcoming events")
        return upcoming
    
    def load_race_session(self, year: int, event: str) -> Optional[fastf1.core.Session]:
        """
        Load a race session with comprehensive error handling.
        
        Args:
            year: Season year
            event: Event name or round number
        
        Returns:
            Loaded session object, or None if failed
        """
        try:
            logger.info(f"Attempting to load: {year} {event} Race")
            session = fastf1.get_session(year, event, 'R')
            
            # Load with minimal options to maximize success rate
            # Telemetry not needed for outcome prediction
            session.load(laps=True, telemetry=False, weather=False)
            
            # Validate that we got actual data
            if session.results is None or len(session.results) == 0:
                logger.warning(f"No results data for {year} {event}")
                return None
            
            if session.laps is None or len(session.laps) == 0:
                logger.warning(f"No laps data for {year} {event}")
                return None
            
            logger.info(f"✓ Loaded {year} {event}: {len(session.results)} drivers")
            return session
            
        except Exception as e:
            logger.warning(f"✗ Failed to load {year} {event}: {str(e)[:100]}")
            return None
    
    def extract_race_results(self, session: fastf1.core.Session) -> Optional[pd.DataFrame]:
        """
        Extract race results from a loaded session.
        
        Returns DataFrame with columns:
        - DriverCode: Driver abbreviation (VER, HAM, etc.)
        - TeamName: Team name (as reported by FastF1)
        - GridPosition: Starting grid position
        - Position: Final classified position
        - Status: Race status (Finished, +1 Lap, Retired, etc.)
        - Points: Championship points earned
        """
        try:
            results = session.results.copy()
            
            # Extract key columns
            df = pd.DataFrame({
                'DriverCode': results['Abbreviation'],
                'DriverNumber': results['DriverNumber'],
                'TeamName': results['TeamName'],
                'GridPosition': results['GridPosition'],
                'Position': results['Position'],
                'Status': results['Status'],
                'Points': results['Points'],
            })
            
            # Handle missing grid positions (e.g., pit lane starts)
            # FastF1 sometimes returns 0 for missing data
            df['GridPosition'] = df['GridPosition'].replace(0, np.nan)
            
            # Convert positions to numeric (handles 'NC' for not classified)
            df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract results: {e}")
            return None
    
    def is_wet_race(self, session: fastf1.core.Session) -> bool:
        """
        Determine if race was wet (simple heuristic).
        
        LIMITATION: This is a rough approximation. Ideally would check:
        - Weather data (if available)
        - Tire compound usage across laps
        - Race control messages about track conditions
        
        For now: Check if intermediate/wet tires were used significantly.
        """
        try:
            if session.laps is None or len(session.laps) == 0:
                return False
            
            # Count tire compound usage
            compounds = session.laps['Compound'].value_counts()
            wet_laps = compounds.get('INTERMEDIATE', 0) + compounds.get('WET', 0)
            total_laps = len(session.laps)
            
            # If >20% of laps on wet tires, classify as wet race
            wet_threshold = 0.20
            is_wet = (wet_laps / max(total_laps, 1)) > wet_threshold
            
            if is_wet:
                logger.info(f"  → Classified as WET race ({wet_laps}/{total_laps} wet laps)")
            
            return is_wet
            
        except Exception as e:
            logger.warning(f"Could not determine wet race status: {e}")
            return False  # Default to dry
    
    def get_circuit_info(self, session: fastf1.core.Session) -> Dict[str, any]:
        """
        Extract circuit information from session.
        
        Returns dict with:
        - circuit_name: Official circuit name
        - circuit_key: Shortened key (for mapping)
        - round_number: Round number in season
        """
        try:
            event = session.event
            return {
                'circuit_name': event.get('Location', 'Unknown'),
                'circuit_key': event.get('Location', 'Unknown'),
                'round_number': event.get('RoundNumber', 0),
                'event_name': event.get('EventName', 'Unknown'),
            }
        except Exception as e:
            logger.warning(f"Could not extract circuit info: {e}")
            return {
                'circuit_name': 'Unknown',
                'circuit_key': 'Unknown',
                'round_number': 0,
                'event_name': 'Unknown',
            }
    
    def load_all_race_data(self, seasons: List[int], retry_failed: bool = False) -> pd.DataFrame:
        """
        Load all race data for given seasons.

        This is the main data collection function. It:
        1. Iterates through all seasons
        2. Gets completed events for each season
        3. Loads race sessions
        4. Extracts results and metadata
        5. Compiles into single DataFrame

        Returns:
            DataFrame with all race results and metadata
            Columns: Year, EventName, CircuitName, RoundNumber, IsWetRace,
                     DriverCode, TeamName, GridPosition, Position, etc.
        """
        all_data = []

        for year in seasons:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading data for {year} season")
            logger.info(f"{'='*60}")

            events = self.get_completed_events(year)

            for event in events:
                # Skip if we're not retrying and this race failed before
                if not retry_failed:
                    race_key = self.tracker.get_race_key(year, event)
                    if race_key in self.tracker.status['races']:
                        if self.tracker.status['races'][race_key]['status'] == 'failed':
                            logger.info(f"Skipping previously failed race: {year} {event}")
                            continue

                # Mark attempt in tracker
                self.tracker.mark_attempt(year, event)

                session = self.load_race_session(year, event)

                if session is None:
                    error_msg = f"Failed to load session data for {year} {event}"
                    self.tracker.mark_failure(year, event, error_msg)
                    continue

                # Extract results
                results = self.extract_race_results(session)
                if results is None:
                    error_msg = f"Failed to extract results for {year} {event}"
                    self.tracker.mark_failure(year, event, error_msg)
                    continue

                # Add metadata
                circuit_info = self.get_circuit_info(session)
                results['Year'] = year
                results['EventName'] = circuit_info['event_name']
                results['CircuitName'] = circuit_info['circuit_name']
                results['RoundNumber'] = circuit_info['round_number']
                results['IsWetRace'] = int(self.is_wet_race(session))

                all_data.append(results)

                # Mark success in tracker
                self.tracker.mark_success(year, event, len(results))

        # Print summary
        self.tracker.print_summary()
        self.tracker.export_to_csv()

        if not all_data:
            logger.error("No data loaded! Check FastF1 connectivity and data availability.")
            return pd.DataFrame()

        # Combine all races
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL DATA LOADED:")
        logger.info(f"  Races: {len(df['EventName'].unique())}")
        logger.info(f"  Drivers: {len(df['DriverCode'].unique())}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"{'='*60}\n")

        return df


# ==========================
# UTILITY FUNCTIONS
# ==========================

def save_raw_data(df: pd.DataFrame, filepath: str = 'raw_race_data.csv'):
    """Save raw race data to CSV for backup/inspection."""
    df.to_csv(filepath, index=False)
    logger.info(f"Saved raw data to {filepath}")


def load_raw_data(filepath: str = 'raw_race_data.csv') -> pd.DataFrame:
    """Load previously saved raw data (faster than re-fetching)."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw data from {filepath}: {len(df)} rows")
    return df


# ==========================
# EXAMPLE USAGE
# ==========================

if __name__ == '__main__':
    # Demo: Load data for training seasons
    loader = F1DataLoader()
    
    # Load 2022-2024 seasons (2025 might be incomplete/problematic)
    df = loader.load_all_race_data([2022, 2023, 2024])
    
    if not df.empty:
        print("\nSample data:")
        print(df.head(10))
        
        print("\nData summary:")
        print(df.describe())
        
        # Save for future use
        save_raw_data(df)
    else:
        print("\nFAILED: Could not load any data. Check:")
        print("1. Internet connection")
        print("2. FastF1 library version (pip install --upgrade fastf1)")
        print("3. F1 API availability")