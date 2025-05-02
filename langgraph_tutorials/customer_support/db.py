"""Database utilities for the customer support application.

Handles downloading, initializing, and managing the SQLite database,
with a 'dirty' state that updates flight dates to current time.

This implementation is used for the tutorial purposes only and is not
meant for production use.
"""

import os
import shutil
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import requests


class DatabaseManager:
    """Manages the customer support database with original and dirty states."""

    DEFAULT_DB_URL = (
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    )
    ORIGINAL_FILE = "travel_original.sqlite"
    DIRTY_FILE = "travel_dirty.sqlite"

    def __init__(self, db_url: str = DEFAULT_DB_URL) -> None:
        self.db_url: str = db_url
        self.original_file: str = self.ORIGINAL_FILE
        self.dirty_file: str = self.DIRTY_FILE

    def initialize(self, *, force_download: bool = False) -> str:
        """Ensure 'original' database is present and create 'dirty' copy with updated dates.

        Args:
            force_download: Force re-download of the original database.

        Returns:
            Path to the dirty database.
        """
        if force_download or not os.path.exists(self.original_file):
            self._download_original()
        self._reset_dirty_with_updated_dates()
        return self.dirty_file

    def _download_original(self) -> None:
        """Download the original database from the remote URL."""
        print("Downloading original database...")
        response: requests.Response = requests.get(self.db_url, timeout=10)
        response.raise_for_status()
        with open(self.original_file, "wb") as f:
            f.write(response.content)

    def _reset_dirty_with_updated_dates(self) -> None:
        """Create a fresh dirty database by copying and updating the original."""
        shutil.copy(self.original_file, self.dirty_file)
        self._update_dates(self.dirty_file)

    def _update_dates(self, db_path: str) -> None:
        """Update flight and booking dates in the database to match current time."""
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        tables: list[str] = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        ).name.tolist()
        dataframes: dict[str, pd.DataFrame] = {
            t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in tables
        }

        flights: pd.DataFrame | None = dataframes.get("flights")
        if flights is not None:
            example_time = pd.to_datetime(
                flights["actual_departure"].replace("\\N", pd.NaT)
            ).max()
            current_time = pd.Timestamp.now(tz=example_time.tz)
            time_diff = current_time - example_time

            for col in [
                "scheduled_departure",
                "scheduled_arrival",
                "actual_departure",
                "actual_arrival",
            ]:
                flights[col] = (
                    pd.to_datetime(flights[col].replace("\\N", pd.NaT)) + time_diff
                )

        bookings: pd.DataFrame | None = dataframes.get("bookings")
        if bookings is not None:
            bookings["book_date"] = (
                pd.to_datetime(bookings["book_date"].replace("\\N", pd.NaT), utc=True)
                + time_diff
            )

        for table_name, df in dataframes.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        conn.commit()
        conn.close()

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager to provide a SQLite cursor for the dirty database.

        Raises:
            FileNotFoundError: If the dirty database file is missing.

        Yields:
            sqlite3.Cursor: A database cursor for performing operations.
        """
        # File check is done here to provide a helpful failure message to
        # potential users. This is not meant to be production code, but
        # rather a tutorial example.
        if not os.path.exists(self.dirty_file):
            msg = (
                f"The dirty database file '{self.dirty_file}' does not exist.\n"
                f"Please run 'manager.initialize()' first to set up the database."
            )
            raise FileNotFoundError(msg)
        conn: sqlite3.Connection = sqlite3.connect(self.dirty_file)
        cursor: sqlite3.Cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        finally:
            cursor.close()
            conn.close()


# Global manager instance
DB: DatabaseManager = DatabaseManager()
