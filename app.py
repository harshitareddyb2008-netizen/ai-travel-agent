"""
AI Travel Analyst - Streamlit Dashboard
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
st.title("✈️ AI Travel Analyst")
st.write(
    "Explore flight prices, estimate a fare for your own trip, and see the "
    "factors that are associated with higher or lower flight prices."
)

tab_overview, tab_estimator, tab_factors, tab_insights = st.tabs(
    ["📊 Overview", "💰 Price Estimator", "🔍 Price Factors", "💡 Insights & Recommendations"]
)

airline_options = ["All"] + sorted(df["Airline"].dropna().unique())
source_options = ["All"] + sorted(df["Source"].dropna().unique())
class_options = ["All"] + sorted(df["Travel_Class"].dropna().unique())
stops_options = ["All"] + sorted(df["Total_Stops"].dropna().unique())

# ============================================================
# TAB 1: OVERVIEW - filters, KPIs, and the core charts
# ============================================================
with tab_overview:
    st.sidebar.header("Filters")
    selected_airline = st.sidebar.selectbox("Airline", airline_options, key="filter_airline")
    selected_source = st.sidebar.selectbox("Source City", source_options, key="filter_source")
    selected_stops = st.sidebar.selectbox("Number of Stops", stops_options, key="filter_stops")

    filtered = df.copy()
    if selected_airline != "All":
        filtered = filtered[filtered["Airline"] == selected_airline]
    if selected_source != "All":
        filtered = filtered[filtered["Source"] == selected_source]
    if selected_stops != "All":
        filtered = filtered[filtered["Total_Stops"] == selected_stops]

    st.subheader("Key Numbers")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Price", f"₹{filtered['Price'].mean():,.0f}")
    col2.metric("Minimum Price", f"₹{filtered['Price'].min():,.0f}")
    col3.metric("Maximum Price", f"₹{filtered['Price'].max():,.0f}")
    col4.metric("Number of Flights", f"{len(filtered):,}")

    st.download_button(
        "⬇ Download filtered flights as CSV",
        filtered.to_csv(index=False),
        "filtered_flights.csv",
        "text/csv",
    )

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

# ============================================================
# TAB 2: PRICE ESTIMATOR - pick a trip, see what similar flights cost
# ============================================================
with tab_estimator:
    st.subheader("Estimate a Fare")
    st.write(
        "Pick your trip details below. This looks up similar flights already "
        "in the data and averages their price - it is not a machine learning "
        "prediction, just a simple, explainable lookup."
    )

    e1, e2, e3, e4 = st.columns(4)
    est_airline = e1.selectbox("Airline", airline_options, key="est_airline")
    est_source = e2.selectbox("Source City", source_options, key="est_source")
    est_class = e3.selectbox("Travel Class", class_options, key="est_class")
    est_stops = e4.selectbox("Stops", stops_options, key="est_stops")

    match = df.copy()
    if est_airline != "All":
        match = match[match["Airline"] == est_airline]
    if est_source != "All":
        match = match[match["Source"] == est_source]
    if est_class != "All":
        match = match[match["Travel_Class"] == est_class]
    if est_stops != "All":
        match = match[match["Total_Stops"] == est_stops]

    if len(match) == 0:
        st.warning("No flights in the data match that exact combination - try loosening a filter.")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated Average Price", f"₹{match['Price'].mean():,.0f}")
        r2.metric("Typical Range (25th-75th %)", f"₹{match['Price'].quantile(.25):,.0f} - ₹{match['Price'].quantile(.75):,.0f}")
        r3.metric("Based on", f"{len(match):,} similar flights")

# ============================================================
# TAB 3: PRICE FACTORS - correlation heatmap + a factor table
# computed live from the data (not hand-typed numbers)
# ============================================================
with tab_factors:
    st.subheader("Which Factors Move the Price?")

    numeric_cols = ["Price", "Distance_km", "Duration_Hours", "Days_Before_Departure", "Total_Stops"]
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Numeric Factors")
    st.pyplot(fig)
    st.caption("A value close to 1 or -1 means a strong relationship; close to 0 means a weak one.")

    avg_by_airline = df.groupby("Airline")["Price"].mean()
    avg_by_source = df.groupby("Source")["Price"].mean()

    factor_table = pd.DataFrame({
        "Factor": ["Travel Class", "Distance", "Airline", "Duration", "Source City", "Stops", "Days Before Departure"],
        "Observation (computed live from the data)": [
            f"Economy avg ₹{df[df['Travel_Class'] == 'Economy']['Price'].mean():,.0f} -> "
            f"First avg ₹{df[df['Travel_Class'] == 'First']['Price'].mean():,.0f}",
            f"Correlation with price: {df['Distance_km'].corr(df['Price']):.2f}",
            f"Cheapest airline avg ₹{avg_by_airline.min():,.0f}, priciest avg ₹{avg_by_airline.max():,.0f}",
            f"Correlation with price: {df['Duration_Hours'].corr(df['Price']):.2f}",
            f"Highest-price source avg ₹{avg_by_source.max():,.0f}, lowest avg ₹{avg_by_source.min():,.0f}",
            f"0 stops avg ₹{df[df['Total_Stops'] == 0]['Price'].mean():,.0f} -> "
            f"2 stops avg ₹{df[df['Total_Stops'] == 2]['Price'].mean():,.0f}",
            f"Correlation with price: {df['Days_Before_Departure'].corr(df['Price']):.2f} (very weak)",
        ],
    })
    st.dataframe(factor_table, hide_index=True, use_container_width=True)
    st.caption("These are observed associations in the data, not proof of cause and effect.")

# ============================================================
# TAB 4: INSIGHTS & RECOMMENDATIONS
# ============================================================
with tab_insights:
    st.subheader("Key Insights")
    st.markdown(
        """
- Prices are right-skewed: the average price is higher than the median, because a
  smaller number of expensive international flights pull the average up.
- Travel class is one of the clearest price factors: price rises consistently from
  Economy to Premium Economy to Business to First.
- Airlines fall into three price tiers, largely tied to whether they fly
  domestic/budget routes or international routes.
- Distance and duration are almost perfectly correlated with *each other* (0.99),
  so they largely capture the same underlying effect - a longer trip.
- Flights with more stops tend to have a *higher* average price in this data,
  likely because stops are more common on long international routes.
- Days before departure has almost no relationship with price here (correlation
  -0.10) - booking early doesn't clearly save money in this dataset.
- The month of departure has only a small effect on average price compared to
  airline, class, or distance.
"""
    )

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
- Don't rely on booking far in advance to save money here - it showed almost no
  effect on price, unlike class, airline, or distance.
"""
    )
