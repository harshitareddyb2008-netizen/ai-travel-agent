"""
AI Travel Analyst - Simple Streamlit Dashboard
Run with: streamlit run app.py

This file cleans the raw dataset itself (same simple steps as analysis.py)
so the dashboard works standalone when deployed - it doesn't depend on
analysis.py having been run first.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="AI Travel Analyst", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/flight_pricing_dataset.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=["Price"])

    # Clean text out of numeric columns, e.g. "Rs. 200,000.00", "150 km", "3 days", "two".
    df["Price"] = df["Price"].astype(str).str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False)
    df["Distance_km"] = df["Distance_km"].astype(str).str.replace(" km", "", regex=False)
    df["Days_Before_Departure"] = df["Days_Before_Departure"].astype(str).str.replace(" days", "", regex=False)
    word_to_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    df["Passenger_Count"] = df["Passenger_Count"].replace(word_to_number)
    for col in ["Price", "Distance_km", "Days_Before_Departure", "Passenger_Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Price"])

    # Standardize airline spelling.
    df["Airline"] = df["Airline"].str.strip().str.title()

    # Standardize city names (name / "name Airport" / airport code -> one name).
    airport_codes = {
        "AMD": "Ahmedabad", "BLR": "Bangalore", "BKK": "Bangkok", "MAA": "Chennai",
        "DEL": "Delhi", "DOH": "Doha", "DXB": "Dubai", "FRA": "Frankfurt",
        "GOI": "Goa", "HYD": "Hyderabad", "JAI": "Jaipur", "CCU": "Kolkata",
        "LHR": "London", "BOM": "Mumbai", "JFK": "New York", "PNQ": "Pune",
        "SIN": "Singapore", "SYD": "Sydney",
    }
    for col in ["Source", "Destination"]:
        df[col] = df[col].str.replace(" Airport", "", regex=False).str.strip()
        df[col] = df[col].replace(airport_codes)

    # Turn stops text into a plain number.
    stop_map = {"non-stop": 0, "0": 0, "1 stop": 1, "1": 1, "2 stops": 2, "2": 2}
    df["Total_Stops"] = df["Total_Stops"].map(stop_map)

    # Turn duration text into hours.
    def to_hours(value):
        text = str(value)
        if "h" in text:
            hours, minutes = text.replace("h", "").replace("m", "").split()
            return round(int(hours) + int(minutes) / 60, 2)
        if "min" in text:
            return round(float(text.replace("min", "").strip()) / 60, 2)
        return float(text)

    df["Duration_Hours"] = df["Duration"].dropna().apply(to_hours)

    return df


df = load_data()

# ---------- Header ----------
st.title("AI Travel Analyst")
st.write(
    "Explore flight prices, compare airlines and routes, and see the "
    "factors that are associated with higher or lower flight prices."
)

# ---------- Filters (sidebar) ----------
st.sidebar.header("Filters")

airline_options = ["All"] + sorted(df["Airline"].dropna().unique())
source_options = ["All"] + sorted(df["Source"].dropna().unique())
stops_options = ["All"] + sorted(df["Total_Stops"].dropna().unique())

selected_airline = st.sidebar.selectbox("Airline", airline_options)
selected_source = st.sidebar.selectbox("Source City", source_options)
selected_stops = st.sidebar.selectbox("Number of Stops", stops_options)

filtered = df.copy()
if selected_airline != "All":
    filtered = filtered[filtered["Airline"] == selected_airline]
if selected_source != "All":
    filtered = filtered[filtered["Source"] == selected_source]
if selected_stops != "All":
    filtered = filtered[filtered["Total_Stops"] == selected_stops]

# ---------- KPIs ----------
st.subheader("Key Numbers")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Price", f"₹{filtered['Price'].mean():,.0f}")
col2.metric("Minimum Price", f"₹{filtered['Price'].min():,.0f}")
col3.metric("Maximum Price", f"₹{filtered['Price'].max():,.0f}")
col4.metric("Number of Flights", f"{len(filtered):,}")

# ---------- Charts ----------
st.subheader("Charts")
sns.set_style("whitegrid")

chart1, chart2 = st.columns(2)

with chart1:
    fig, ax = plt.subplots()
    sns.histplot(filtered["Price"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Flight Prices")
    st.pyplot(fig)

with chart2:
    fig, ax = plt.subplots()
    avg_by_stops = filtered.groupby("Total_Stops")["Price"].mean()
    sns.barplot(x=avg_by_stops.index, y=avg_by_stops.values, ax=ax)
    ax.set_title("Average Price by Number of Stops")
    st.pyplot(fig)

chart3, chart4 = st.columns(2)

with chart3:
    fig, ax = plt.subplots()
    avg_by_class = filtered.groupby("Travel_Class")["Price"].mean().sort_values()
    sns.barplot(x=avg_by_class.index, y=avg_by_class.values, ax=ax)
    ax.set_title("Average Price by Travel Class")
    st.pyplot(fig)

with chart4:
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered, x="Duration_Hours", y="Price", alpha=0.3, ax=ax)
    ax.set_title("Price vs Flight Duration")
    st.pyplot(fig)

# ---------- Insights ----------
st.subheader("Key Insights")
st.markdown(
    """
- Prices are right-skewed: the average price is higher than the median, because a
  smaller number of expensive international flights pull the average up.
- Travel class is one of the clearest price factors: price rises consistently from
  Economy to Premium Economy to Business to First.
- Airlines fall into three price tiers, largely tied to whether they fly
  domestic/budget routes or international routes.
- Flight distance and duration are both positively associated with price
  (longer flights tend to cost more).
- Flights with more stops tend to have a *higher* average price in this data,
  likely because stops are more common on long international routes.
- The month of departure has only a small effect on average price compared to
  airline, class, or distance.
"""
)

# ---------- Recommendations ----------
st.subheader("Recommendations for Travelers")
st.markdown(
    """
- Consider budget-tier airlines when the route allows it, since they show much
  lower average prices than full-service international carriers.
- Don't assume a flight with stops will be cheaper - in this data, stops are
  associated with *higher* prices, not lower.
- If your dates are flexible, Economy or Premium Economy class offers a large
  saving over Business or First.
- Expect route/distance to be a major cost driver - long-haul international
  routes will generally cost much more than short domestic ones.
"""
)
