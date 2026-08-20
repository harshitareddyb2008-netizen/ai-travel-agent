"""
AI Travel Analyst - Part 1
Simple data cleaning, EDA and visualizations for the flight price dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# STEP 1: LOAD AND INSPECT THE DATA
# ======================================================

df = pd.read_csv("data/flight_pricing_dataset.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ======================================================
# STEP 2: DATA CLEANING
# ======================================================

# Problem: Exact duplicate rows exist in the data.
# Why: Duplicates don't add new information and can skew averages.
df = df.drop_duplicates()

# Problem: Some rows have no Price at all.
# Why: Price is what we are analyzing, so a row without it is useless to us.
df = df.dropna(subset=["Price"])

# Problem: Numeric columns have extra text mixed in, e.g. "Rs. 200,000.00",
# "150 km", "3 days", or the word "two" instead of the number 2.
# Why: We need real numbers to do maths and plot graphs.
df["Price"] = df["Price"].astype(str).str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False)
df["Distance_km"] = df["Distance_km"].astype(str).str.replace(" km", "", regex=False)
df["Days_Before_Departure"] = df["Days_Before_Departure"].astype(str).str.replace(" days", "", regex=False)

word_to_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
df["Passenger_Count"] = df["Passenger_Count"].replace(word_to_number)

for col in ["Price", "Distance_km", "Days_Before_Departure", "Passenger_Count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Now that Price is a real number, drop the few rows that still failed to convert.
df = df.dropna(subset=["Price"])

# Problem: The same airline appears with different capitalization
# (e.g. "INDIGO", "indigo", "Indigo").
# Why: These should all be counted as one airline, not three.
df["Airline"] = df["Airline"].str.strip().str.title()

# Problem: Source/Destination cities are written 3 different ways:
# a city name ("Mumbai"), a city name + "Airport" ("Mumbai Airport"),
# or a 3-letter airport code ("BOM").
# Why: All three mean the same city and should be grouped together.
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

# Problem: Total_Stops mixes text ("non-stop", "1 stop") and numbers ("0", "1").
# Why: We need a plain number of stops to compare against price.
stop_map = {"non-stop": 0, "0": 0, "1 stop": 1, "1": 1, "2 stops": 2, "2": 2}
df["Total_Stops"] = df["Total_Stops"].map(stop_map)

# Problem: Duration is written in 3 different formats:
# "4h 00m", "339 min", or a plain decimal like "6.71".
# Why: We need one consistent number (hours) to compare against price.
def to_hours(value):
    text = str(value)
    if "h" in text:
        hours, minutes = text.replace("h", "").replace("m", "").split()
        return round(int(hours) + int(minutes) / 60, 2)
    if "min" in text:
        return round(float(text.replace("min", "").strip()) / 60, 2)
    return float(text)

df["Duration_Hours"] = df["Duration"].dropna().apply(to_hours)

# Problem: Departure_Date is stored as plain text.
# Why: We need it as a real date so we can pull out the month.
df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")
df["Month"] = df["Departure_Date"].dt.month_name()

print("\nShape after cleaning:", df.shape)

# Save the cleaned data so app.py can reuse it.
df.to_csv("cleaned_flight_data.csv", index=False)

# ======================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ======================================================

print("\n--- Price ---")
print(df["Price"].describe())

print("\n--- Average price by airline ---")
print(df.groupby("Airline")["Price"].mean().sort_values())

print("\n--- Average price by number of stops ---")
print(df.groupby("Total_Stops")["Price"].mean())

print("\n--- Correlation: Duration vs Price ---")
print(df["Duration_Hours"].corr(df["Price"]))

print("\n--- Correlation: Distance vs Price ---")
print(df["Distance_km"].corr(df["Price"]))

print("\n--- Correlation: Days Before Departure vs Price ---")
print(df["Days_Before_Departure"].corr(df["Price"]))

print("\n--- Average price by source city ---")
print(df.groupby("Source")["Price"].mean().sort_values(ascending=False))

print("\n--- Average price by travel class ---")
print(df.groupby("Travel_Class")["Price"].mean().sort_values())

print("\n--- Average price by month ---")
print(df.groupby("Month")["Price"].mean())

# ======================================================
# STEP 4: VISUALIZATIONS
# ======================================================

sns.set_style("whitegrid")

# Graph 1: Distribution of flight prices
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=40, kde=True)
plt.title("Distribution of Flight Prices")
plt.xlabel("Price")
plt.savefig("screenshots/1_price_distribution.png", bbox_inches="tight")
plt.show()

# Graph 2: Average price by airline
plt.figure(figsize=(10, 5))
avg_by_airline = df.groupby("Airline")["Price"].mean().sort_values()
sns.barplot(x=avg_by_airline.index, y=avg_by_airline.values)
plt.xticks(rotation=45, ha="right")
plt.title("Average Flight Price by Airline")
plt.ylabel("Average Price")
plt.savefig("screenshots/2_price_by_airline.png", bbox_inches="tight")
plt.show()

# Graph 3: Price vs number of stops
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Total_Stops", y="Price")
plt.title("Price vs Number of Stops")
plt.savefig("screenshots/3_price_vs_stops.png", bbox_inches="tight")
plt.show()

# Graph 4: Price vs flight duration
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Duration_Hours", y="Price", alpha=0.3)
plt.title("Price vs Flight Duration")
plt.xlabel("Duration (hours)")
plt.savefig("screenshots/4_price_vs_duration.png", bbox_inches="tight")
plt.show()

# Graph 5: Average price by source city
plt.figure(figsize=(10, 5))
avg_by_source = df.groupby("Source")["Price"].mean().sort_values(ascending=False)
sns.barplot(x=avg_by_source.index, y=avg_by_source.values)
plt.xticks(rotation=45, ha="right")
plt.title("Average Flight Price by Source City")
plt.ylabel("Average Price")
plt.savefig("screenshots/5_price_by_source.png", bbox_inches="tight")
plt.show()

# Bonus Graph 6: Average price by travel class
plt.figure(figsize=(7, 5))
avg_by_class = df.groupby("Travel_Class")["Price"].mean().sort_values()
sns.barplot(x=avg_by_class.index, y=avg_by_class.values)
plt.title("Average Flight Price by Travel Class")
plt.ylabel("Average Price")
plt.savefig("screenshots/6_price_by_class.png", bbox_inches="tight")
plt.show()

# Bonus Graph 7: Correlation heatmap - all numeric factors at a glance
plt.figure(figsize=(7, 5))
numeric_cols = ["Price", "Distance_km", "Duration_Hours", "Days_Before_Departure", "Total_Stops"]
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Factors")
plt.savefig("screenshots/7_correlation_heatmap.png", bbox_inches="tight")
plt.show()

print("\nAll charts saved in the screenshots folder.")
