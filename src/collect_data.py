import fastf1
import pandas as pd
import os


# create cache folder
if not os.path.exists("cache"):
    os.makedirs("cache")


fastf1.Cache.enable_cache("cache")


year = 2024
race = "Monaco"


session = fastf1.get_session(
    year,
    race,
    "R"
)


session.load()


laps = session.laps


data = laps[
    [
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "TyreLife"
    ]
]


data = data.dropna()


data["LapTime"] = (
    data["LapTime"]
    .dt.total_seconds()
)


data.to_csv(
    "data/f1_data.csv",
    index=False
)


print("RaceOS dataset created!")