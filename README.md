# AI Travel Analyst

Part 1 (Exploration) submission for the MIC AIML Department Recruitment Challenge — Track 3: Data Science & Visualization.

## 1. Project Overview

This project analyzes a flight price dataset to understand what actually drives flight prices. It cleans the raw data, explores it with simple Pandas operations, builds 6 visualizations, identifies the major price factors, and turns those findings into insights and traveler recommendations. A small Streamlit dashboard presents the same analysis interactively.

## 2. Problem Statement

Flight prices look unpredictable to a traveler booking a ticket. The goal of this project is to explore a real flight price dataset and answer: which factors (airline, stops, duration, route, class, month) are actually associated with higher or lower prices, and what should a traveler take away from that?

## 3. Dataset

- File: `data/flight_pricing_dataset.csv` (kept in a `data/` folder so the repo root stays clean)
- Rows: 100,000 | Columns: 18
- Columns: `Flight_ID, Airline, Source, Destination, Departure_Date, Departure_Time, Arrival_Time, Duration, Total_Stops, Distance_km, Travel_Class, Days_Before_Departure, Season, Weekday, Aircraft_Type, Booking_Channel, Passenger_Count, Price`
- The dataset is intentionally messy: every column has ~5% missing values, ~2,000 duplicate rows exist, and several columns store the same information in more than one format (explained below).

## 4. Data Cleaning

Each step below follows **Problem → Code → Why**.

**Duplicate rows** — 1,961 exact duplicate rows existed.
```python
df = df.drop_duplicates()
```
Duplicates don't add new information and would double-count some flights.

**Missing Price** — some rows had no price at all.
```python
df = df.dropna(subset=["Price"])
```
Price is what we're analyzing, so a row without it can't be used.

**Numbers stored as text with extra symbols** — `Price` had values like `"Rs. 200,000.00"`, `Distance_km` had `"150 km"`, `Days_Before_Departure` had `"3 days"`, and `Passenger_Count` had words like `"two"` instead of `2`.
```python
df["Price"] = df["Price"].astype(str).str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False)
df["Distance_km"] = df["Distance_km"].astype(str).str.replace(" km", "", regex=False)
df["Days_Before_Departure"] = df["Days_Before_Departure"].astype(str).str.replace(" days", "", regex=False)
word_to_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
df["Passenger_Count"] = df["Passenger_Count"].replace(word_to_number)
for col in ["Price", "Distance_km", "Days_Before_Departure", "Passenger_Count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```
We need real numbers to do maths and plot graphs, not text.

**Inconsistent airline spelling** — `"INDIGO"`, `"indigo"`, `"Indigo"` were 3 different strings for one airline.
```python
df["Airline"] = df["Airline"].str.strip().str.title()
```
The same airline shouldn't be split into multiple categories.

**Inconsistent city names** — Source/Destination appeared as a city name (`"Mumbai"`), a city name + "Airport" (`"Mumbai Airport"`), or a 3-letter airport code (`"BOM"`).
```python
airport_codes = {"BOM": "Mumbai", "DEL": "Delhi", ...}  # 18 codes total
df[col] = df[col].str.replace(" Airport", "", regex=False).str.strip().replace(airport_codes)
```
All three forms mean the same city and should be grouped together for a fair comparison.

**Mixed stop formats** — `Total_Stops` mixed `"non-stop"`/`"0"`, `"1 stop"`/`"1"`, `"2 stops"`/`"2"`.
```python
stop_map = {"non-stop": 0, "0": 0, "1 stop": 1, "1": 1, "2 stops": 2, "2": 2}
df["Total_Stops"] = df["Total_Stops"].map(stop_map)
```
We need a plain number of stops to compare against price.

**Mixed duration formats** — `Duration` mixed `"4h 00m"`, `"339 min"`, and plain decimals like `"6.71"`.
```python
def to_hours(value):
    text = str(value)
    if "h" in text:
        hours, minutes = text.replace("h", "").replace("m", "").split()
        return round(int(hours) + int(minutes) / 60, 2)
    if "min" in text:
        return round(float(text.replace("min", "").strip()) / 60, 2)
    return float(text)
```
One consistent unit (hours) is needed to compare duration against price.

**Date as text** — `Departure_Date` was plain text.
```python
df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")
df["Month"] = df["Departure_Date"].dt.month_name()
```
We need a real date to pull out the month.

Missing values in other columns were **left as NaN**, not filled in — with ~93,000 usable rows there's enough data left, and guessing values for the columns whose effect on price we're measuring would bias the results.

**Result:** 100,000 → 93,083 rows after cleaning.

## 5. Exploratory Data Analysis

| Question | Answer |
|---|---|
| Average price | ₹72,990 |
| Cheapest price | ₹152.13 |
| Most expensive price | ₹999,306.03 |
| Median price | ₹49,100 |
| Cheapest airline (avg) | GoFirst — ₹10,253 |
| Most expensive airline (avg) | Qatar Airways — ₹101,165 |
| Price vs stops | 0 stops ₹61,603 → 1 stop ₹79,447 → 2 stops ₹84,661 (rises with stops) |
| Price vs duration | Correlation = 0.67 (moderately strong, positive) |
| Price vs distance | Correlation = 0.69 (moderately strong, positive) |
| Highest-price source city | New York — ₹155,396 |
| Lowest-price source city | Goa — ₹47,221 |
| Price vs travel class | Economy ₹59,668 → Premium Economy ₹81,857 → Business ₹115,828 → First ₹133,873 |
| Cheapest month | August — ₹68,453 |
| Most expensive month | March — ₹77,976 |

A data quality note found during EDA: about **9.8% of all flights (9,112 rows) are priced at exactly ₹200,000**, which is far more than a real fare distribution would naturally produce at one exact value. This looks like a capped/placeholder value from how the dataset was generated, rather than a genuine fare — flagged here rather than silently ignored (see *Challenges Faced*).

## 6. Visualizations

All charts are in `screenshots/` and are produced by `analysis.py`.

1. **Distribution of Flight Prices** (histogram) — shows the price distribution is right-skewed with a large cluster of cheap flights and a long tail of expensive ones.
2. **Average Price by Airline** (bar chart) — shows 3 clear airline price tiers.
3. **Price vs Number of Stops** (box plot) — shows price rising with more stops.
4. **Price vs Flight Duration** (scatter plot) — shows a positive relationship between duration and price.
5. **Average Price by Source City** (bar chart) — shows international source cities priced far above domestic ones.
6. **Average Price by Travel Class** (bar chart, bonus) — shows a clean step-up in price with each class.

## 7. Major Factors Affecting Flight Prices

| Factor | Observation |
|---|---|
| Travel Class | Strongest, cleanest factor — average price rises consistently from Economy to First |
| Distance | Longer distance is associated with a higher price (correlation 0.69) |
| Airline | Airlines fall into 3 price tiers, associated with whether they run budget/domestic or international routes |
| Duration | Longer duration is associated with a higher price (correlation 0.67) |
| Source City | International source cities show much higher average prices than domestic cities |
| Stops | More stops is associated with a *higher* average price, likely because stops are more common on long international routes |
| Month | Only a small effect — about 12% difference between the cheapest and priciest month |

None of these factors are claimed to *cause* price changes — they are associations observed in this dataset.

## 8. Key Insights

1. The average price (₹72,990) is well above the median (₹49,100) — the distribution is right-skewed, with a smaller number of expensive international flights pulling the average up.
2. Airlines split into three clear price tiers: budget carriers (GoFirst, Indigo, AirAsia India, SpiceJet) average ₹10,000–11,000, Vistara and Air India average around ₹45,000, and international carriers (Qatar Airways, Singapore Airlines, Emirates, etc.) average ₹98,000–101,000.
3. Flights with more stops tend to have a *higher* average price, not lower — this is the opposite of common intuition, and most likely happens because long international routes are the ones that need stopovers.
4. Flight duration and distance both show a moderately strong positive relationship with price (correlations of 0.67 and 0.69), making them two of the strongest price drivers found.
5. Source city matters a lot: flights from New York (₹155,396) and Sydney (₹142,725) average roughly 3x the price of flights from Indian cities like Goa (₹47,221) or Hyderabad (₹48,309).
6. Travel class is one of the clearest price factors — Economy, Premium Economy, Business, and First each step up in price in that exact order.
7. The month of departure has only a mild effect on price — a ~12% gap between the cheapest (August) and priciest (March) months, far smaller than the swing seen across airlines or classes.
8. About 9.8% of flights are priced at exactly ₹200,000, an unusually large spike for one exact value — worth flagging as a likely data artifact rather than a real pricing pattern.

## 9. Recommendations

- Travelers looking for lower fares may consider budget-tier airlines (GoFirst, Indigo, AirAsia India, SpiceJet) over full-service carriers when the route allows it.
- Don't assume a route with stops will be cheaper — in this data, more stops is associated with higher prices, not lower, so it's worth comparing actual fares rather than assuming.
- Travelers with flexible plans could consider Economy or Premium Economy, since Business and First carry substantially higher average prices.
- Because price rises with distance and duration, travelers flying long-haul international routes (e.g. from New York, Sydney, London) should budget for a much higher fare than domestic routes.
- Since month-to-month differences are relatively small in this data, choice of airline and travel class is a bigger lever for savings than trying to time the month of travel.

## 10. Technologies Used

- **Python** — programming language
- **Pandas** — loading, cleaning, and analyzing the data
- **Matplotlib & Seaborn** — visualizations
- **Streamlit** — the interactive dashboard

No machine learning is used in Part 1 — all findings come from simple grouping, averaging, and correlation.

## 11. How to Run

```bash
pip install -r requirements.txt

# Run the analysis (cleans the data, prints EDA, saves charts to screenshots/)
python analysis.py

# Run the dashboard (uses the cleaned data saved by analysis.py)
streamlit run app.py
```
Run `analysis.py` first — it creates `cleaned_flight_data.csv`, which `app.py` reads.

## 12. Future Improvements

- Part 2 could add a simple regression model to predict price and compare it against these observed factors.
- The exact-₹200,000 spike could be investigated further (e.g. by checking if it correlates with a specific Season, Booking_Channel, or Aircraft_Type in the raw data).
- More granular geography (e.g. actual routes, not just source city) could sharpen the location-based insights.

---

## Interview Preparation

**Why did you choose this project?**
I chose Track 3 (Data Science & Visualization) because I wanted to practice the full data analysis workflow — cleaning messy real-world-style data, exploring it, and turning numbers into a story — using tools I actually understand well: Pandas, Matplotlib, and Seaborn.

**What is your dataset?**
It's the official flight pricing dataset provided for this challenge — 100,000 flight records with 18 columns covering airline, route, timing, duration, stops, class, and price.

**How did you clean the data?**
I checked `.isnull().sum()` and `.duplicated().sum()` first to see what was actually wrong, then fixed only real problems: dropped 1,961 duplicate rows, dropped rows with no price, converted number-like text columns (which had things like "Rs. 200,000.00", "150 km", and "two") into real numbers, standardized airline names and city names that were written in multiple ways, and converted stops and duration into consistent numeric formats.

**Why did you remove duplicates?**
Identical rows don't add new information — keeping them would double-count some flights and slightly bias any average I calculate.

**How did you handle missing values?**
I only dropped rows where the value I actually needed for a specific analysis was missing (like Price). I didn't fill in guessed values for other columns, because inventing numbers for the very columns I'm using to measure price effects would bias my own results.

**Why did you choose these six visualizations?**
Each one answers a specific question from the challenge brief: overall price distribution, price by airline, price by stops, price vs duration, price by source city, and (as a bonus) price by travel class, which turned out to be the strongest factor.

**What is the strongest factor affecting price?**
Travel class showed the cleanest, most consistent pattern — price increases at every step from Economy to Premium Economy to Business to First. Distance and duration were also strong (correlation around 0.67–0.69).

**What is correlation?**
It's a number between -1 and 1 that measures how strongly two numeric variables move together. Close to 1 means they rise together, close to -1 means one rises as the other falls, and close to 0 means little relationship.

**Does correlation mean causation?**
No. A correlation just shows that two things move together in the data — it doesn't prove one causes the other. For example, stops being associated with higher prices doesn't mean adding a stop makes a flight more expensive; it's more likely that both are driven by something else, like the flight being a long international route.

**Why did you use Pandas?**
It's the standard, simple tool for tabular data in Python — reading CSVs, cleaning columns, and grouping data all have short, readable one-line solutions.

**Why did you use Seaborn?**
It builds on Matplotlib but needs less code for common statistical charts like box plots and histograms, and its default styling is cleaner.

**Why did you use Streamlit?**
It turns a Python script into an interactive dashboard with very little code — no separate frontend needed — which fits the "keep it simple" goal of this project.

**What was the biggest challenge?**
Figuring out exactly how each column was messy before writing any cleaning code. For example, I initially converted Price straight to numbers and lost about 3% of rows — inspecting *why* they failed showed they were formatted like "Rs. 200,000.00", so I fixed the actual text problem instead of just dropping the rows.

**What would you improve in the future?**
I'd dig deeper into the exact-₹200,000 price spike I found, and in Part 2 I'd build a simple prediction model to check whether it agrees with the factors I found through basic EDA here.
