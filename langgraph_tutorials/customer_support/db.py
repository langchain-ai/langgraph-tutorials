"""Database utilities for the customer support application.

This module handles database initialization, updates, and maintenance for the
customer support system. It downloads a SQLite database if needed and provides
functionality to update flight dates to current time.
"""

import os
import shutil
import sqlite3

import pandas as pd
import requests

# Database configuration
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False

# Download database if needed
if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)


def update_dates(file):
    """Updates flight and booking dates to current time.

    Creates a fresh copy of the database from backup and updates all datetime
    fields to be relative to the current date, maintaining the same relative
    time differences from the original data.

    Args:
        file (str): Path to the database file to update

    Returns:
        str: Path to the updated database file

    Note:
        This function modifies the following tables:
        - flights: Updates scheduled/actual departure/arrival times
        - bookings: Updates booking dates
    """
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    return file


db = update_dates(local_file)
