import re
from datetime import datetime
from typing import List


def is_date_in_week(date_str: str, week_str: str) -> bool:
    """
    Checks if the given date (DD-MM-YYYY) falls within the given week string (e.g. "week 30, 2025").
    Returns True if yes, False otherwise.
    """
    if date_str is None or week_str is None:
        return False
    # Parse the date string
    date_obj = datetime.strptime(date_str, "%d-%m-%Y")
    date_iso_year, date_iso_week, _ = date_obj.isocalendar()

    # Extract week number and year from the week string (flexible format)
    match = re.search(r"week\s*?(\d{1,2})\D*?(\d{4})", week_str, re.IGNORECASE)
    if not match:
        # raise ValueError(f"Could not parse week string: {week_str}")
        return False

    week_num = int(match.group(1))
    week_year = int(match.group(2))

    return (date_iso_year == week_year) and (date_iso_week == week_num)


def any_date_in_week(date_list: List[str], week_str: str) -> bool:
    """
    Checks if any date in the given list (DD-MM-YYYY) falls within the given week string
    (e.g. "week 30, 2025").
    Returns True if at least one date matches, False otherwise.
    """
    if not date_list or week_str is None:
        return False

    # Extract week number and year from the week string
    match = re.search(r"week\s*?(\d{1,2})\D*?(\d{4})", week_str, re.IGNORECASE)
    if not match:
        return False

    week_num = int(match.group(1))
    week_year = int(match.group(2))

    for date_str in date_list:
        if not date_str:
            continue
        try:
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")
            date_iso_year, date_iso_week, _ = date_obj.isocalendar()
            if date_iso_year == week_year and date_iso_week == week_num:
                return True
        except ValueError:
            continue

    return False
