import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the data from CSV
df_cost_revenue = pd.read_csv("cost_revenue_dirty.csv")

# Extract basic information
df_rows_columns = df_cost_revenue.shape
nan_rows_sum = df_cost_revenue.isna().values.any()
duplicated_rows = df_cost_revenue.duplicated().values.any()
columns_data_type = df_cost_revenue.dtypes

print("Information about dataset:\n")
print(f"Rows: {df_rows_columns[0]}, Columns: {df_rows_columns[1]}\n")
print(f"Nan values: {nan_rows_sum}\n")
print(f"Duplicated data: {duplicated_rows}\n")
print(f"Type of data:\n {columns_data_type}")

# Remove dollar signs and commas, then convert to numeric type
df_cost_revenue['USD_Production_Budget'] = df_cost_revenue['USD_Production_Budget'].str.replace("$", "").str.replace(
    ",", "")
df_cost_revenue['USD_Worldwide_Gross'] = df_cost_revenue['USD_Worldwide_Gross'].str.replace("$", "").str.replace(",",
                                                                                                                 "")
df_cost_revenue['USD_Domestic_Gross'] = df_cost_revenue['USD_Domestic_Gross'].str.replace("$", "").str.replace(",", "")

# Convert cleaned strings to float type for numerical calculations
df_cost_revenue['USD_Production_Budget'] = pd.to_numeric(df_cost_revenue['USD_Production_Budget'], errors='coerce')
df_cost_revenue['USD_Worldwide_Gross'] = pd.to_numeric(df_cost_revenue['USD_Worldwide_Gross'], errors='coerce')
df_cost_revenue['USD_Domestic_Gross'] = pd.to_numeric(df_cost_revenue['USD_Domestic_Gross'], errors='coerce')
df_cost_revenue['Release_Date'] = pd.to_datetime(df_cost_revenue['Release_Date'], errors='coerce')

# Display the updated data types
print("\nUpdated data types:\n")
print(df_cost_revenue.dtypes)

# Clean data with excluding unreleased film in time of data collection
scrape_date = pd.Timestamp('2018-5-1')
future_release = df_cost_revenue[df_cost_revenue.Release_Date >= scrape_date]
df_cost_revenue = df_cost_revenue.drop(future_release.index)

# Calculate insights about the data
data_insights = df_cost_revenue[
    ['USD_Production_Budget', 'USD_Worldwide_Gross', 'USD_Domestic_Gross']].describe().apply(
    lambda x: x.apply('{0:.2f}'.format))
print(f"\nData Insights:\n{data_insights}")

# Count films that grossed exactly 0 domestically and worldwide
domestic_gross_zero = (df_cost_revenue['USD_Domestic_Gross'] == 0).sum()
world_wide_zero = (df_cost_revenue['USD_Worldwide_Gross'] == 0).sum()

print(f"\nFilms that grossed 0$ domestically: {domestic_gross_zero}")
print(f"Films that grossed 0$ worldwide: {world_wide_zero}")

# Find the biggest flop: highest production budget with 0 domestic gross
biggest_flop = df_cost_revenue[(df_cost_revenue['USD_Domestic_Gross'] == 0)].nlargest(1, 'USD_Production_Budget')

print(f"\nBiggest flop:\n{biggest_flop.to_string()}")

# Plot scatterplot
plt.figure(figsize=(8, 4), dpi=400)
ax = sns.scatterplot(data=df_cost_revenue, x='Release_Date', y='USD_Production_Budget', hue='USD_Worldwide_Gross',
                     size='USD_Worldwide_Gross')
ax.set(xlim=(df_cost_revenue.Release_Date.min(), df_cost_revenue.Release_Date.max()), ylim=(0, 450000000),
       xlabel='Year', ylabel='Budget in $100 millions', )

plt.show()

# Scatter plot and linear regression
plt.figure(figsize=(8, 4), dpi=400)
bx = sns.regplot(data=df_cost_revenue, x='USD_Production_Budget', y='USD_Worldwide_Gross', scatter_kws={'alpha': 0.3},
                 line_kws={'color': '#ff7c43'}, color='#2f4b6c')

bx.set(xlim=(df_cost_revenue.USD_Production_Budget.min(), df_cost_revenue.USD_Production_Budget.max()),
       ylim=(df_cost_revenue.USD_Worldwide_Gross.min(), df_cost_revenue.USD_Worldwide_Gross.max()),
       xlabel='Budget in $100 millions', ylabel='Revenue in $ billions')

plt.show()

# Linear Regression using Scikit-learn
regression = LinearRegression()
X = pd.DataFrame(df_cost_revenue, columns=['USD_Production_Budget'])
y = pd.DataFrame(df_cost_revenue, columns=['USD_Worldwide_Gross'])

regression.fit(X, y)

# Theta zero
print("Regression intercept:", regression.intercept_)

# Theta one
print("Regression coef:", regression.coef_)

# R-squared
print("Regression score:", regression.score(X, y))

# Check what will be the revenue for film with xxx budget?
budget = 400000000
revenue_estimation = round(regression.intercept_[0] + regression.coef_[0,0] * budget, 2)
print(revenue_estimation)
