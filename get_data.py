import fastf1
import pandas as pd

# Enable cache (so it doesn't download everything repeatedly)
fastf1.Cache.enable_cache("cache")

# Load a race session
session = fastf1.get_session(2024, "British", "R")

print("Loading F1 data...")
session.load()

# Get lap data
laps = session.laps

print(laps.head())

# Save data
laps.to_csv("data/race_data.csv", index=False)

print("Saved race data!")