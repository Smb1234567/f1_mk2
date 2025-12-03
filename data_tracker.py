"""
F1 Race Predictor - Data Loading Status Tracker
Tracks which races are successfully loaded and which failed.
Enables retrying failed loads and ensures complete data coverage.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceLoadingTracker:
    """
    Tracks the status of race data loading attempts.

    Maintains a JSON file with:
    - Which races were attempted
    - Success/failure status
    - Error messages for failures
    - Timestamp of last attempt
    """

    def __init__(self, tracking_file: str = 'race_loading_status.json'):
        self.tracking_file = tracking_file
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load existing tracking data or create new."""
        if Path(self.tracking_file).exists():
            with open(self.tracking_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded tracking data: {len(data.get('races', {}))} races tracked")
                return data
        else:
            logger.info("No existing tracking data, starting fresh")
            return {
                'races': {},
                'last_updated': None,
                'total_attempts': 0,
                'total_successes': 0,
                'total_failures': 0
            }

    def _save_status(self):
        """Save tracking data to disk."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def get_race_key(self, year: int, event: str) -> str:
        """Generate unique key for a race."""
        return f"{year}_{event.replace(' ', '_')}"

    def mark_attempt(self, year: int, event: str):
        """Mark that we're attempting to load this race."""
        key = self.get_race_key(year, event)

        if key not in self.status['races']:
            self.status['races'][key] = {
                'year': year,
                'event': event,
                'status': 'attempting',
                'attempts': 0,
                'last_attempt': None,
                'last_success': None,
                'error_message': None
            }

        self.status['races'][key]['attempts'] += 1
        self.status['races'][key]['last_attempt'] = datetime.now().isoformat()
        self.status['races'][key]['status'] = 'attempting'
        self.status['total_attempts'] += 1

        self._save_status()

    def mark_success(self, year: int, event: str, num_drivers: int):
        """Mark successful load."""
        key = self.get_race_key(year, event)

        if key in self.status['races']:
            self.status['races'][key]['status'] = 'success'
            self.status['races'][key]['last_success'] = datetime.now().isoformat()
            self.status['races'][key]['num_drivers'] = num_drivers
            self.status['races'][key]['error_message'] = None

            # Update totals (only count first success)
            if self.status['races'][key]['attempts'] == 1:
                self.status['total_successes'] += 1

        self.status['last_updated'] = datetime.now().isoformat()
        self._save_status()

    def mark_failure(self, year: int, event: str, error_message: str):
        """Mark failed load with error message."""
        key = self.get_race_key(year, event)

        if key in self.status['races']:
            self.status['races'][key]['status'] = 'failed'
            self.status['races'][key]['error_message'] = error_message[:200]  # Limit length

            # Only count as failure if all attempts failed
            self.status['total_failures'] += 1

        self.status['last_updated'] = datetime.now().isoformat()
        self._save_status()

    def get_failed_races(self) -> List[Tuple[int, str]]:
        """Get list of races that failed to load."""
        failed = []
        for key, data in self.status['races'].items():
            if data['status'] == 'failed':
                failed.append((data['year'], data['event']))
        return failed

    def get_successful_races(self) -> List[Tuple[int, str]]:
        """Get list of races that successfully loaded."""
        successful = []
        for key, data in self.status['races'].items():
            if data['status'] == 'success':
                successful.append((data['year'], data['event']))
        return successful

    def get_never_attempted(self, expected_races: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """Get races that were never attempted."""
        attempted_keys = set(self.status['races'].keys())
        never_attempted = []

        for year, event in expected_races:
            key = self.get_race_key(year, event)
            if key not in attempted_keys:
                never_attempted.append((year, event))

        return never_attempted

    def print_summary(self):
        """Print comprehensive summary of loading status."""
        logger.info("\n" + "="*70)
        logger.info("RACE DATA LOADING SUMMARY")
        logger.info("="*70)

        successful = [r for r in self.status['races'].values() if r['status'] == 'success']
        failed = [r for r in self.status['races'].values() if r['status'] == 'failed']
        attempting = [r for r in self.status['races'].values() if r['status'] == 'attempting']

        logger.info(f"Total Attempts:      {self.status['total_attempts']}")
        logger.info(f"Successful Loads:    {len(successful)}")
        logger.info(f"Failed Loads:        {len(failed)}")
        logger.info(f"Currently Attempting: {len(attempting)}")

        if failed:
            logger.info("\n" + "-"*70)
            logger.info("FAILED RACES (need retry):")
            logger.info("-"*70)
            for race in sorted(failed, key=lambda x: (x['year'], x['event'])):
                logger.info(f"  {race['year']} {race['event']}")
                logger.info(f"    Error: {race['error_message']}")
                logger.info(f"    Attempts: {race['attempts']}")

        if successful:
            logger.info("\n" + "-"*70)
            logger.info("SUCCESSFUL RACES:")
            logger.info("-"*70)
            by_year = {}
            for race in successful:
                year = race['year']
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(race['event'])

            for year in sorted(by_year.keys()):
                logger.info(f"  {year}: {len(by_year[year])} races")
                for event in by_year[year]:
                    logger.info(f"    ✓ {event}")

        logger.info("="*70 + "\n")

    def export_to_csv(self, filepath: str = 'race_loading_report.csv'):
        """Export tracking data to CSV for easy inspection."""
        rows = []
        for key, data in self.status['races'].items():
            rows.append({
                'Year': data['year'],
                'Event': data['event'],
                'Status': data['status'],
                'Attempts': data['attempts'],
                'LastAttempt': data['last_attempt'],
                'LastSuccess': data['last_success'],
                'NumDrivers': data.get('num_drivers', None),
                'ErrorMessage': data['error_message']
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(['Year', 'Event'])
        df.to_csv(filepath, index=False)
        logger.info(f"Exported tracking report to {filepath}")
        return df

    def reset_failed_races(self):
        """Reset failed races so they can be retried."""
        count = 0
        for key, data in self.status['races'].items():
            if data['status'] == 'failed':
                data['status'] = 'ready_for_retry'
                data['error_message'] = None
                count += 1

        self._save_status()
        logger.info(f"Reset {count} failed races for retry")
        return count


def get_expected_races_for_seasons(seasons: List[int]) -> Dict[int, List[str]]:
    """
    Get expected race events for given seasons.

    IMPORTANT: This is a manual mapping of known races per season.
    It serves as ground truth to compare against what FastF1 returns.

    Returns:
        Dict mapping year to list of expected event names
    """
    expected_races = {
        2022: [
            'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix',
            'Emilia Romagna Grand Prix', 'Miami Grand Prix', 'Spanish Grand Prix',
            'Monaco Grand Prix', 'Azerbaijan Grand Prix', 'Canadian Grand Prix',
            'British Grand Prix', 'Austrian Grand Prix', 'French Grand Prix',
            'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix',
            'Italian Grand Prix', 'Singapore Grand Prix', 'Japanese Grand Prix',
            'United States Grand Prix', 'Mexico City Grand Prix',
            'São Paulo Grand Prix', 'Abu Dhabi Grand Prix'
        ],
        2023: [
            'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix',
            'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix',
            'Spanish Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix',
            'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix',
            'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix',
            'Japanese Grand Prix', 'Qatar Grand Prix', 'United States Grand Prix',
            'Mexico City Grand Prix', 'São Paulo Grand Prix', 'Las Vegas Grand Prix',
            'Abu Dhabi Grand Prix'
        ],
        2024: [
            'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix',
            'Japanese Grand Prix', 'Chinese Grand Prix', 'Miami Grand Prix',
            'Emilia Romagna Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix',
            'Spanish Grand Prix', 'Austrian Grand Prix', 'British Grand Prix',
            'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix',
            'Italian Grand Prix', 'Azerbaijan Grand Prix', 'Singapore Grand Prix',
            'United States Grand Prix', 'Mexico City Grand Prix', 'São Paulo Grand Prix',
            'Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix'
        ]
    }

    # Filter to requested seasons
    return {year: races for year, races in expected_races.items() if year in seasons}


def validate_data_completeness(tracker: RaceLoadingTracker, seasons: List[int]) -> Tuple[bool, List[str]]:
    """
    Validate that we have complete data coverage for given seasons.

    Returns:
        (is_complete, list_of_issues)
    """
    expected_races = get_expected_races_for_seasons(seasons)
    issues = []

    for year, events in expected_races.items():
        logger.info(f"\nValidating {year} season ({len(events)} expected races)...")

        for event in events:
            key = tracker.get_race_key(year, event)

            if key not in tracker.status['races']:
                issues.append(f"{year} {event}: Never attempted")
                logger.warning(f"  ✗ {event}: Not attempted")
            elif tracker.status['races'][key]['status'] == 'failed':
                issues.append(f"{year} {event}: Failed to load")
                logger.warning(f"  ✗ {event}: Load failed")
            elif tracker.status['races'][key]['status'] == 'success':
                logger.info(f"  ✓ {event}: OK")
            else:
                issues.append(f"{year} {event}: Status unknown")
                logger.warning(f"  ? {event}: Unknown status")

    is_complete = len(issues) == 0

    if is_complete:
        logger.info("\n✓ Data coverage is COMPLETE for all seasons")
    else:
        logger.warning(f"\n✗ Data coverage is INCOMPLETE: {len(issues)} issues found")

    return is_complete, issues


# ==========================
# EXAMPLE USAGE
# ==========================

if __name__ == '__main__':
    """
    Test the tracking system.
    """
    tracker = RaceLoadingTracker()

    # Simulate some loading attempts
    tracker.mark_attempt(2022, 'Bahrain')
    tracker.mark_success(2022, 'Bahrain', 20)

    tracker.mark_attempt(2022, 'Saudi Arabia')
    tracker.mark_failure(2022, 'Saudi Arabia', 'Session data not available')

    tracker.mark_attempt(2023, 'Monaco')
    tracker.mark_success(2023, 'Monaco', 20)

    # Print summary
    tracker.print_summary()

    # Export to CSV
    df = tracker.export_to_csv()
    print("\nTracking DataFrame:")
    print(df)

    # Validate completeness
    is_complete, issues = validate_data_completeness(tracker, [2022, 2023, 2024])

    if issues:
        print("\n\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")