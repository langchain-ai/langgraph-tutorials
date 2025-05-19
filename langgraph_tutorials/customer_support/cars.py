from datetime import date, datetime

from langchain_core.tools import tool

from langgraph_tutorials.customer_support.db import DB


@tool
def search_car_rentals(
    location: str | None = None,
    name: str | None = None,
    price_tier: str | None = None,
    start_date: datetime | date | None = None,
    end_date: datetime | date | None = None,
) -> list[dict]:
    """Search for car rentals based on location, name, price tier, start date, and end date.

    Args:
        location (Optional[str]): The location of the car rental.
        name (Optional[str]): The name of the car rental company.
        price_tier (Optional[str]): The price tier of the car rental.
        start_date (Optional[Union[datetime, date]]): The start date of the car rental.
        end_date (Optional[Union[datetime, date]]): The end date of the car rental.

    Returns:
        list[dict]: A list of car rental dictionaries matching the search criteria.
    """
    with DB.get_cursor() as cursor:
        query = "SELECT * FROM car_rentals WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For our tutorial, we will let you match on any dates and price tier.
        # (since our toy dataset doesn't have much data)
        cursor.execute(query, params)
        results = cursor.fetchall()

        return [
            dict(zip([column[0] for column in cursor.description], row, strict=False))
            for row in results
        ]


@tool
def book_car_rental(rental_id: int) -> str:
    """Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    with DB.get_cursor() as cursor:
        cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
        if cursor.rowcount > 0:
            return f"Car rental {rental_id} successfully booked."
        return f"No car rental found with ID {rental_id}."


@tool
def update_car_rental(
    rental_id: int,
    start_date: datetime | date | None = None,
    end_date: datetime | date | None = None,
) -> str:
    """Update a car rental's start and end dates by its ID.

    Args:
        rental_id (int): The ID of the car rental to update.
        start_date (Optional[Union[datetime, date]]): The new start date of the car rental.
        end_date (Optional[Union[datetime, date]]): The new end date of the car rental.

    Returns:
        str: A message indicating whether the car rental was successfully updated or not.
    """
    with DB.get_cursor() as cursor:
        updates_made = 0
        if start_date:
            cursor.execute(
                "UPDATE car_rentals SET start_date = ? WHERE id = ?",
                (start_date, rental_id),
            )
            updates_made += cursor.rowcount
        if end_date:
            cursor.execute(
                "UPDATE car_rentals SET end_date = ? WHERE id = ?",
                (end_date, rental_id),
            )
            updates_made += cursor.rowcount

        if updates_made > 0:
            return f"Car rental {rental_id} successfully updated."
        return f"No car rental found with ID {rental_id}."

@tool
def cancel_car_rental(rental_id: int) -> str:
    """Cancel a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to cancel.

    Returns:
        str: A message indicating whether the car rental was successfully cancelled or not.
    """
    with DB.get_cursor() as cursor:
        cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
        if cursor.rowcount > 0:
            return f"Car rental {rental_id} successfully cancelled."
        return f"No car rental found with ID {rental_id}."
