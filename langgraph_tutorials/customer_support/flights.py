from datetime import date, datetime

import pytz
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph_tutorials.customer_support.db import DB


@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user, including flight details and seat assignments."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id")
    if not passenger_id:
        msg = "No passenger ID configured."
        raise ValueError(msg)

    with DB.get_cursor() as cursor:
        query = """
        SELECT
            t.ticket_no, t.book_ref,
            f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
            bp.seat_no, tf.fare_conditions
        FROM
            tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
        WHERE
            t.passenger_id = ?
        """
        cursor.execute(query, (passenger_id,))
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        return [dict(zip(column_names, row, strict=False)) for row in rows]


@tool
def search_flights(
    departure_airport: str | None = None,
    arrival_airport: str | None = None,
    start_time: date | datetime | None = None,
    end_time: date | datetime | None = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    with DB.get_cursor() as cursor:
        query = "SELECT * FROM flights WHERE 1 = 1"
        params = []

        if departure_airport:
            query += " AND departure_airport = ?"
            params.append(departure_airport)

        if arrival_airport:
            query += " AND arrival_airport = ?"
            params.append(arrival_airport)

        if start_time:
            query += " AND scheduled_departure >= ?"
            params.append(start_time)

        if end_time:
            query += " AND scheduled_departure <= ?"
            params.append(end_time)

        query += " LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        return [dict(zip(column_names, row, strict=False)) for row in rows]


@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id")
    if not passenger_id:
        msg = "No passenger ID configured."
        raise ValueError(msg)

    with DB.get_cursor() as cursor:
        cursor.execute(
            "SELECT departure_airport, arrival_airport, scheduled_departure "
            "FROM flights "
            "WHERE flight_id = ?",
            (new_flight_id,),
        )
        new_flight = cursor.fetchone()
        if not new_flight:
            return "Invalid new flight ID provided."

        column_names = [column[0] for column in cursor.description]
        new_flight_dict = dict(zip(column_names, new_flight, strict=False))
        timezone = pytz.timezone("Etc/GMT-3")
        current_time = datetime.now(tz=timezone)
        departure_time = datetime.strptime(
            new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
        )
        time_until = (departure_time - current_time).total_seconds()
        if time_until < (3 * 3600):
            return (
                f"Not permitted to reschedule to a flight less than 3 hours away. "
                f"Selected flight departs at {departure_time}."
            )

        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
        current_flight = cursor.fetchone()
        if not current_flight:
            return "No existing ticket found for the given ticket number."

        cursor.execute(
            "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        current_ticket = cursor.fetchone()
        if not current_ticket:
            return f"Passenger ID {passenger_id} does not own ticket {ticket_no}."

        cursor.execute(
            "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
            (new_flight_id, ticket_no),
        )

    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id")
    if not passenger_id:
        msg = "No passenger ID configured."
        raise ValueError(msg)

    with DB.get_cursor() as cursor:
        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
        existing_ticket = cursor.fetchone()
        if not existing_ticket:
            return "No existing ticket found for the given ticket number."

        cursor.execute(
            "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        current_ticket = cursor.fetchone()
        if not current_ticket:
            return f"Passenger ID {passenger_id} does not own ticket {ticket_no}."

        cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))

    return "Ticket successfully cancelled."

