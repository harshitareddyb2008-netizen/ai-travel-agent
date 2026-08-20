"""
AI Travel Analyst - Simple Streamlit Dashboard
Run with: streamlit run app.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="AI Travel Analyst", layout="wide")

# ---------- Load cleaned data ----------
# analysis.py already cleaned the raw data and saved this file.
df = pd.read_csv("cleaned_flight_data.csv")

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
