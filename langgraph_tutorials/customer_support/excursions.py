"""Information about excursions"""


from langgraph_tutorials.customer_support.db import DB


def search_trip_recommendations(
    location: str | None = None,
    name: str | None = None,
    keywords: str | None = None,
) -> list[dict]:
    """Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation.
        name (Optional[str]): The name of the trip recommendation.
        keywords (Optional[str]): The keywords associated with the trip recommendation.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    with DB.get_cursor() as cursor:
        query = "SELECT * FROM trip_recommendations WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        if keywords:
            keyword_list = keywords.split(",")
            keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
            query += f" AND ({keyword_conditions})"
            params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

        cursor.execute(query, params)
        results = cursor.fetchall()

        return [
            dict(zip([column[0] for column in cursor.description], row, strict=False))
            for row in results
        ]


def book_excursion(recommendation_id: int) -> str:
    """Book an excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    with DB.get_cursor() as cursor:
        cursor.execute(
            "UPDATE trip_recommendations SET booked = 1 WHERE id = ?",
            (recommendation_id,),
        )
        if cursor.rowcount > 0:
            return f"Trip recommendation {recommendation_id} successfully booked."
        return f"No trip recommendation found with ID {recommendation_id}."


def update_excursion(recommendation_id: int, details: str) -> str:
    """Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    with DB.get_cursor() as cursor:
        cursor.execute(
            "UPDATE trip_recommendations SET details = ? WHERE id = ?",
            (details, recommendation_id),
        )
        if cursor.rowcount > 0:
            return f"Trip recommendation {recommendation_id} successfully updated."
        return f"No trip recommendation found with ID {recommendation_id}."


def cancel_excursion(recommendation_id: int) -> str:
    """Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    with DB.get_cursor() as cursor:
        cursor.execute(
            "UPDATE trip_recommendations SET booked = 0 WHERE id = ?",
            (recommendation_id,),
        )
        if cursor.rowcount > 0:
            return f"Trip recommendation {recommendation_id} successfully cancelled."
        return f"No trip recommendation found with ID {recommendation_id}."
