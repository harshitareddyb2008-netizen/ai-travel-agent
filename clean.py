"""
Cleaning functions for the flight pricing dataset.

The raw file is deliberately messy: every column has ~5% missing values, there
are ~2k duplicate rows, and several columns store the same fact in more than
one format (e.g. "4h 00m" and 1.67 both mean four hours).

Each function below fixes exactly one problem, so each can be tested and
explained on its own.
"""

import re
import pandas as pd

# Each city appears as a name, a name + " Airport", or a 3-letter airport code.
# Mapping the codes back to city names collapses 54 spellings down to 18 cities.
AIRPORT_CODES = {
    "AMD": "Ahmedabad",  "BLR": "Bangalore", "BKK": "Bangkok",
    "MAA": "Chennai",    "DEL": "Delhi",     "DOH": "Doha",
    "DXB": "Dubai",      "FRA": "Frankfurt", "GOI": "Goa",
    "HYD": "Hyderabad",  "JAI": "Jaipur",    "CCU": "Kolkata",
    "LHR": "London",     "BOM": "Mumbai",    "JFK": "New York",
    "PNQ": "Pune",       "SIN": "Singapore", "SYD": "Sydney",
}


def standardise_airline(series):
    """'AIR INDIA', 'air india' and 'Air India' are one airline, not three.

    Title-casing makes the capitalisation consistent, which turns 39 distinct
    spellings into 13 real airlines.
    """
    return series.str.strip().str.title()


def standardise_city(series):
    """Reduce 'Mumbai', 'Mumbai Airport' and 'BOM' to a single name: 'Mumbai'."""
    names = series.str.strip().str.replace(r"\s+Airport$", "", regex=True)
    return names.replace(AIRPORT_CODES)


def parse_stops(value):
    """'non-stop' and '0' mean the same thing. Return the number of stops.

    Pulling the first digit out of the text handles '1 stop' and '2 stops',
    and 'non-stop' has no digit at all, so it maps to 0.
    """
    if pd.isna(value):
        return None
    found = re.search(r"\d", str(value))
    return int(found.group()) if found else 0


def parse_duration(value):
    """Return flight duration in hours as a number.

    Two formats appear in the file: '4h 00m' (about 82k rows) and a plain
    decimal like 1.67 (about 12k rows). Both are converted to decimal hours so
    the column can be used in maths and charts.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(\d+)\s*h\s*(\d+)?", text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2) or 0)
        return round(hours + minutes / 60, 2)
    try:
        return float(text)
    except ValueError:
        return None


def parse_hour(value):
    """Return the hour of day (0-23) from a time written in 12h or 24h form.

    The file mixes '8:10 PM' with '21:50'. Only the hour is kept, because the
    analysis groups flights into departure hours rather than exact minutes.
    """
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    match = re.match(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    if "PM" in text and hour != 12:
        hour += 12
    if "AM" in text and hour == 12:
        hour = 0
    return hour


# Columns that are stored as text in the raw file but are really numbers.
NUMERIC_COLUMNS = ["Distance_km", "Days_Before_Departure", "Passenger_Count", "Price"]


def clean_dataset(df):
    """Run every cleaning step and return the tidy DataFrame.

    Missing values are left as NaN rather than filled in. With 100k rows there
    is plenty of data left after dropping the unusable ones, and inventing
    values for the very columns whose effect on price is being measured would
    bias the result.
    """
    df = df.copy()

    # 1. Identical rows carry no extra information and would double-count.
    df = df.drop_duplicates()

    # 2. Price is the value being explained, so a row without it is unusable.
    df = df.dropna(subset=["Price"])

    # 3. Make the spelling of names consistent.
    df["Airline"] = standardise_airline(df["Airline"])
    df["Source"] = standardise_city(df["Source"])
    df["Destination"] = standardise_city(df["Destination"])

    # 4. Convert the mixed-format columns into plain numbers.
    df["Total_Stops"] = df["Total_Stops"].map(parse_stops)
    df["Duration_Hours"] = df["Duration"].map(parse_duration)
    df["Departure_Hour"] = df["Departure_Time"].map(parse_hour)

    # 5. Columns that look like text but hold numbers.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")

    # 6. A flight cannot leave and arrive at the same moment; drop impossible rows.
    df = df[(df["Duration_Hours"].isna()) | (df["Duration_Hours"] > 0)]

    return df
