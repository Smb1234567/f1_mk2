"""
Sample data for 2025 F1 season drivers and teams
This file provides a reliable sample dataset for predictions without relying on FastF1 API
"""

# 2025 confirmed driver roster based on official announcements
SAMPLE_2025_DRIVERS = {
    'VER': 'Red Bull Racing',        # Max Verstappen
    'PER': 'Red Bull Racing',        # Sergio Perez (assuming he stays)
    'LEC': 'Ferrari',                # Charles Leclerc
    'SAI': 'Ferrari',                # Carlos Sainz (assuming he stays)
    'NOR': 'McLaren',                # Lando Norris
    'PIA': 'McLaren',                # Oscar Piastri
    'RUS': 'Mercedes',               # George Russell
    'HAM': 'Mercedes',               # Lewis Hamilton (assuming he stays)
    'ALO': 'Aston Martin',           # Fernando Alonso (assuming he stays)
    'STR': 'Aston Martin',           # Lance Stroll (assuming he stays)
    'GAS': 'Alpine',                 # Pierre Gasly (assuming he stays)
    'OCO': 'Alpine',                 # Esteban Ocon (assuming he stays)
    'DEV': 'Williams',               # Alex Albon
    'SAR': 'Williams',               # Logan Sargeant (or a new driver)
    'MAG': 'Haas',                   # Kevin Magnussen (assuming he stays)
    'HUL': 'Haas',                   # Nico Hulkenberg (assuming he stays)
    'TSU': 'RB',                     # Yuki Tsunoda
    'RIC': 'RB',                     # Daniel Ricciardo (or a new driver)
    'ZHO': 'Kick Sauber',            # Zhou Guanyu (assuming he stays)
    'BOT': 'Kick Sauber',            # Valtteri Bottas (assuming he stays)
}

# Actual confirmed 2025 lineup based on official announcements
ACTUAL_2025_DRIVERS = {
    'VER': 'Red Bull Racing',        # Max Verstappen
    'LAW': 'Red Bull Racing',        # Liam Lawson (confirmed as second driver)
    'LEC': 'Ferrari',                # Charles Leclerc
    'HAM': 'Ferrari',                # Lewis Hamilton (confirmed move from Mercedes)
    'NOR': 'McLaren',                # Lando Norris
    'PIA': 'McLaren',                # Oscar Piastri
    'RUS': 'Mercedes',               # George Russell
    'ANT': 'Mercedes',               # Andrea Kimi Antonelli (confirmed as new driver)
    'ALO': 'Aston Martin',           # Fernando Alonso
    'STR': 'Aston Martin',           # Lance Stroll
    'TSU': 'RB',                     # Yuki Tsunoda
    'HAD': 'RB',                     # Isack Hadjar (confirmed new driver)
    'BEA': 'Haas',                   # Oliver Bearman (confirmed new driver) 
    'OCO': 'Haas',                   # Esteban Ocon (confirmed move from Alpine)
    'GAS': 'Alpine',                 # Pierre Gasly (confirmed)
    'DOO': 'Alpine',                 # Jack Doohan (confirmed new driver)
    'DEV': 'Williams',               # Alex Albon (confirmed)
    'SAI': 'Williams',               # Carlos Sainz (confirmed move from Ferrari)
    'HUL': 'Kick Sauber',            # Nico Hulkenberg (confirmed)
    'BOR': 'Kick Sauber',            # Gabriel Bortoleto (confirmed new driver)
}

def get_2025_sample_drivers():
    """
    Returns the confirmed 2025 F1 driver-team combinations.
    This is the reliable data to use for sample predictions.
    """
    return ACTUAL_2025_DRIVERS

def get_2025_sample_race_grid():
    """
    Returns a sample race grid with drivers, teams, and grid positions for 2025.
    This simulates a qualifying result for quick prediction testing.
    """
    drivers_teams = ACTUAL_2025_DRIVERS
    sample_grid = []
    
    # Create a plausible sample grid for testing purposes
    sample_positions = [
        ('VER', 'Red Bull Racing', 1),    # Max Verstappen - Pole position
        ('LEC', 'Ferrari', 2),            # Charles Leclerc - Second
        ('NOR', 'McLaren', 3),            # Lando Norris - Third
        ('HAM', 'Ferrari', 4),            # Lewis Hamilton - Fourth
        ('RUS', 'Mercedes', 5),           # George Russell - Fifth
        ('PIA', 'McLaren', 6),            # Oscar Piastri - Sixth
        ('LAW', 'Red Bull Racing', 7),    # Liam Lawson - Seventh
        ('ALO', 'Aston Martin', 8),       # Fernando Alonso - Eighth
        ('GAS', 'Alpine', 9),             # Pierre Gasly - Ninth
        ('STR', 'Aston Martin', 10),      # Lance Stroll - Tenth
        ('ANT', 'Mercedes', 11),          # Andrea Kimi Antonelli - Eleventh
        ('TSU', 'RB', 12),                # Yuki Tsunoda - Twelfth
        ('DEV', 'Williams', 13),          # Alex Albon - Thirteenth
        ('HAD', 'RB', 14),                # Isack Hadjar - Fourteenth
        ('BEA', 'Haas', 15),              # Oliver Bearman - Fifteenth
        ('OCO', 'Haas', 16),              # Esteban Ocon - Sixteenth
        ('DOO', 'Alpine', 17),            # Jack Doohan - Seventeenth
        ('SAI', 'Williams', 18),          # Carlos Sainz - Eighteenth
        ('HUL', 'Kick Sauber', 19),       # Nico Hulkenberg - Nineteenth
        ('BOR', 'Kick Sauber', 20),       # Gabriel Bortoleto - Twentieth
    ]
    
    for driver_code, team_name, grid_pos in sample_positions:
        if driver_code in drivers_teams:
            sample_grid.append({
                'DriverCode': driver_code,
                'TeamName': team_name,
                'GridPosition': grid_pos
            })
    
    return sample_grid

def get_driver_names():
    """
    Returns full driver names for the 2025 season
    """
    return {
        'VER': 'Max Verstappen',         # Red Bull
        'LAW': 'Liam Lawson',            # Red Bull
        'LEC': 'Charles Leclerc',        # Ferrari
        'HAM': 'Lewis Hamilton',         # Ferrari
        'NOR': 'Lando Norris',           # McLaren
        'PIA': 'Oscar Piastri',          # McLaren
        'RUS': 'George Russell',         # Mercedes
        'ANT': 'Andrea Kimi Antonelli',  # Mercedes
        'ALO': 'Fernando Alonso',        # Aston Martin
        'STR': 'Lance Stroll',           # Aston Martin
        'TSU': 'Yuki Tsunoda',           # RB
        'HAD': 'Isack Hadjar',          # RB
        'BEA': 'Oliver Bearman',        # Haas
        'OCO': 'Esteban Ocon',          # Haas
        'GAS': 'Pierre Gasly',          # Alpine
        'DOO': 'Jack Doohan',           # Alpine
        'DEV': 'Alex Albon',             # Williams
        'SAI': 'Carlos Sainz',           # Williams
        'HUL': 'Nico Hulkenberg',       # Kick Sauber
        'BOR': 'Gabriel Bortoleto',     # Kick Sauber
    }

if __name__ == '__main__':
    # Test the sample data
    print("2025 Sample Drivers:")
    for code, team in get_2025_sample_drivers().items():
        print(f"  {code}: {team}")
    
    print("\nSample Race Grid:")
    for item in get_2025_sample_race_grid():
        print(f"  Grid {item['GridPosition']}: {item['DriverCode']} ({item['TeamName']})")