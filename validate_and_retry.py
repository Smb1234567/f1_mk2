"""
F1 Race Predictor - Data Validation & Retry Script
Standalone script to check data completeness and retry failed loads
"""

import sys
import argparse
from data_loader import F1DataLoader, save_raw_data
from data_tracker import RaceLoadingTracker, get_expected_races_for_seasons, validate_data_completeness
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Validate F1 race data completeness and retry failed loads'
    )
    parser.add_argument(
        'action',
        choices=['check', 'retry', 'report', 'reset'],
        help='Action to perform: check=validate completeness, retry=retry failed races, report=show loading report, reset=reset all tracking'
    )
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        default=[2022, 2023, 2024],
        help='Seasons to process (default: 2022 2023 2024)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_completeness(args.seasons)
    elif args.action == 'retry':
        retry_failed(args.seasons)
    elif args.action == 'report':
        show_report()
    elif args.action == 'reset':
        reset_tracking()


def check_completeness(seasons):
    """Check data completeness without loading anything."""
    logger.info("="*70)
    logger.info("CHECKING DATA COMPLETENESS")
    logger.info("="*70)
    
    tracker = RaceLoadingTracker()
    
    # Get expected races
    expected_races = get_expected_races_for_seasons(seasons)
    
    logger.info(f"\nExpected race counts:")
    for year, events in expected_races.items():
        logger.info(f"  {year}: {len(events)} races")
    
    # Validate
    is_complete, issues = validate_data_completeness(tracker, seasons)
    
    # Print summary
    tracker.print_summary()
    
    if is_complete:
        logger.info("\n✓ SUCCESS: All races loaded successfully!")
        logger.info("\nYou can proceed with training:")
        logger.info("  python train_model.py")
        return 0
    else:
        logger.warning(f"\n⚠ WARNING: {len(issues)} races missing or failed")
        logger.warning("\nMissing races:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        logger.info("\nTo retry failed races, run:")
        logger.info("  python validate_and_retry.py retry")
        return 1


def retry_failed(seasons):
    """Retry loading failed races."""
    logger.info("="*70)
    logger.info("RETRYING FAILED RACE LOADS")
    logger.info("="*70)
    
    tracker = RaceLoadingTracker()
    failed_races = tracker.get_failed_races()
    
    if not failed_races:
        logger.info("\n✓ No failed races to retry!")
        return 0
    
    logger.info(f"\nFound {len(failed_races)} failed races:")
    for year, event in failed_races:
        logger.info(f"  - {year} {event}")
    
    # Confirm retry
    response = input("\nProceed with retry? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        logger.info("Retry cancelled")
        return 0
    
    # Reset failed races
    tracker.reset_failed_races()
    
    # Load with retry
    loader = F1DataLoader()
    df = loader.load_all_race_data(seasons, retry_failed=True)
    
    if not df.empty:
        save_raw_data(df, 'raw_race_data.csv')
        logger.info("\n✓ Retry complete! Data saved to raw_race_data.csv")
        
        # Check if we're now complete
        is_complete, issues = validate_data_completeness(loader.tracker, seasons)
        if is_complete:
            logger.info("\n✓✓ All races now loaded successfully!")
            return 0
        else:
            logger.warning(f"\n⚠ Still missing {len(issues)} races")
            logger.warning("Some races may be permanently unavailable from FastF1")
            return 1
    else:
        logger.error("\n❌ Retry failed - no new data loaded")
        return 1


def show_report():
    """Show detailed loading report."""
    logger.info("="*70)
    logger.info("RACE LOADING REPORT")
    logger.info("="*70)
    
    tracker = RaceLoadingTracker()
    tracker.print_summary()
    
    # Export to CSV
    df = tracker.export_to_csv('race_loading_report.csv')
    
    print("\n" + "="*70)
    print("Detailed report:")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("Report saved to: race_loading_report.csv")
    print("="*70)
    
    return 0


def reset_tracking():
    """Reset all tracking data (WARNING: destructive)."""
    logger.warning("="*70)
    logger.warning("RESET ALL TRACKING DATA")
    logger.warning("="*70)
    logger.warning("\n⚠ This will erase all race loading history!")
    logger.warning("You will need to reload all data from scratch.")
    
    response = input("\nAre you SURE? (type 'yes' to confirm): ").strip()
    if response != 'yes':
        logger.info("Reset cancelled")
        return 0
    
    import os
    import json
    
    # Reset tracking file
    tracking_file = 'race_loading_status.json'
    if os.path.exists(tracking_file):
        os.remove(tracking_file)
        logger.info(f"✓ Deleted {tracking_file}")
    
    # Create fresh tracker
    tracker = RaceLoadingTracker()
    logger.info("✓ Created fresh tracking file")
    
    logger.info("\nTracking data reset complete!")
    logger.info("Run 'python train_model.py' to reload all data from scratch")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)